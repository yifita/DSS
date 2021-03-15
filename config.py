import yaml
import os
from easydict import EasyDict as edict
import torch
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import FoVPerspectiveCameras
from DSS.core.texture import LightingTexture, NeuralTexture
from DSS.utils import get_class_from_string
from DSS.training.trainer import Trainer
from DSS import logger_py


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    cfg_special = None
    with open(path, 'r') as f:
        cfg_special = edict(yaml.load(f, Loader=yaml.Loader))

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = edict(yaml.load(f, Loader=yaml.Loader))
    else:
        cfg = edict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def save_config(path, config):
    """
    Save config dictionary as json file
    """
    out_dir = os.path.dirname(path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if os.path.isfile(path):
        logger_py.warn(
            "Found file existing in {}, overwriting the existing file.".format(out_dir))

    with open(path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    logger_py.info("Saved config to {}".format(path))


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = edict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def _get_tensor_with_default(opt, key, size, fill_value=0.0):
    if key not in opt:
        return torch.zeros(*size).fill_(fill_value)
    else:
        return torch.FloatTensor(opt[key])


def create_point_texture(opt_renderer_texture):
    from DSS.core.texture import (NeuralTexture, LightingTexture)
    """ create shader that generate per-point color """
    if opt_renderer_texture.texture.is_neural_shader:
        texture = NeuralTexture(opt_renderer_texture.texture)
    else:
        lights = create_lights(opt_renderer_texture.get('lights', None))
        texture = LightingTexture(
            specular=opt_renderer_texture.texture.specular, lights=lights)

    return texture


def create_lights(opt_renderer_texture_lights):
    """
    Create lights specified by opt, if no sun or point lights
    are given, create the tri-color lights.
    Currently only supports the same lights for all batches
    """
    from DSS.core.lighting import (DirectionalLights, PointLights)
    ambient_color = torch.tensor(
        opt_renderer_texture_lights.ambient_color).view(1, -1, 3)
    specular_color = torch.tensor(
        opt_renderer_texture_lights.specular_color).view(1, -1, 3)
    diffuse_color = torch.tensor(
        opt_renderer_texture_lights.diffuse_color).view(1, -1, 3)
    if opt_renderer_texture_lights['type'] == "sun":
        direction = torch.tensor(
            opt_renderer_texture_lights.direction).view(1, -1, 3)
        lights = DirectionalLights(ambient_color=ambient_color, diffuse_color=diffuse_color,
                                   specular_color=specular_color, direction=direction)
    elif opt_renderer_texture_lights['type'] == 'point':
        location = torch.tensor(
            opt_renderer_texture_lights.location).view(1, -1, 3)
        lights = PointLights(ambient_color=ambient_color, diffuse_color=diffuse_color,
                             specular_color=specular_color, location=location)

    return lights


def create_cameras(opt):
    pass


def create_dataset(opt_data, mode="train"):
    import DSS.utils.dataset as DssDataset
    if opt_data.type == 'MVR':
        dataset = DssDataset.MVRDataset(**opt_data, mode=mode)
    elif opt_data.type == 'DTU':
        dataset = DssDataset.DTUDataset(**opt_data, mode=mode)
    else:
        raise NotImplementedError
    return dataset


def create_model(cfg, device, mode="train", camera_model=None, **kwargs):
    ''' Returns model

    Args:
        cfg (edict): imported yaml config
        device (device): pytorch device
    '''
    if cfg.model.type == 'point':
        decoder = None

    texture = None
    use_lighting = (cfg.renderer is not None and not cfg.renderer.get(
                       'is_neural_texture', True))
    if use_lighting:
        texture = LightingTexture()
    else:
        if 'rgb' not in cfg.model.decoder_kwargs.out_dims:
            Texture = get_class_from_string(cfg.model.texture_type)
            cfg.model.texture_kwargs['c_dim'] = cfg.model.decoder_kwargs.out_dims.get('latent', 0)
            texture_decoder = Texture(**cfg.model.texture_kwargs)
        else:
            texture_decoder = decoder
            logger_py.info("Decoder used as NeuralTexture")

        texture = NeuralTexture(
            view_dependent=cfg.model.texture_kwargs.view_dependent, decoder=texture_decoder).to(device=device)
        logger_py.info("Created NeuralTexture {}".format(texture.__class__))
        logger_py.info(texture)

    Model = get_class_from_string(
        "DSS.models.{}_modeling.Model".format(cfg.model.type))

    # if not using decoder, then use non-parameterized point renderer
    # create icosphere as initial point cloud
    sphere_mesh = ico_sphere(level=4)
    sphere_mesh.scale_verts_(0.5)
    points, normals = sample_points_from_meshes(
        sphere_mesh, num_samples=int(
            cfg['model']['model_kwargs']['n_points_per_cloud']),
        return_normals=True)
    colors = torch.ones_like(points)
    renderer = create_renderer(cfg.renderer).to(device)
    model = Model(
        points, normals, colors,
        renderer,
        device=device,
        texture=texture,
        **cfg.model.model_kwargs,
    ).to(device=device)

    return model


def create_generator(cfg, model, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    Generator = get_class_from_string(
        'DSS.models.{}_modeling.Generator'.format(cfg.model.type))

    generator = Generator(model, device,
                          threshold=cfg['test']['threshold'],
                          **cfg.generation)
    return generator


def create_trainer(cfg, model, optimizer, scheduler, generator, train_loader, val_loader, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
        generator (Generator): generator instance to
            generate meshes for visualization
    '''
    threshold = cfg['test']['threshold']
    out_dir = os.path.join(cfg['training']['out_dir'], cfg['name'])
    vis_dir = os.path.join(out_dir, 'vis')
    debug_dir = os.path.join(out_dir, 'debug')
    log_dir = os.path.join(out_dir, 'logs')
    val_dir = os.path.join(out_dir, 'val')

    trainer = Trainer(
        model, optimizer, scheduler, generator, train_loader, val_loader,
        device=device,
        vis_dir=vis_dir, debug_dir=debug_dir, log_dir=log_dir, val_dir=val_dir,
        threshold=threshold,
        **cfg.training)

    return trainer


def create_renderer(render_opt):
    """ Create rendere """
    Renderer = get_class_from_string(render_opt.renderer_type)
    Raster = get_class_from_string(render_opt.raster_type)
    i = render_opt.raster_type.rfind('.')
    raster_setting_type = render_opt.raster_type[:i] + \
        '.PointsRasterizationSettings'
    if render_opt.compositor_type is not None:
        Compositor = get_class_from_string(render_opt.compositor_type)
        compositor = Compositor()
    else:
        compositor = None

    RasterSetting = get_class_from_string(raster_setting_type)
    raster_settings = RasterSetting(**render_opt.raster_params)

    renderer = Renderer(
        rasterizer=Raster(
            cameras=FoVPerspectiveCameras(), raster_settings=raster_settings),
        compositor=compositor,
    )
    return renderer
