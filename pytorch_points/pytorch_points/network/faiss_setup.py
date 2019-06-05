import faiss


GPU_RES = faiss.StandardGpuResources()
# GPU_RES.setTempMemoryFraction(0.1)  # deprecated with faiss 1.5.2
