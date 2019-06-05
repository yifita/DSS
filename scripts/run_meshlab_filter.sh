#!/bin/bash
# usage: ./sample_mesh /mnt/external/points/data/ModelNet40 "*.ply"
num_procs=1

inputDir=$1
name="$2"
outputDir="$3"
myDir=$(pwd)
scriptFile="$myDir/$4"
echo "input: $inputDir output: $outputDir extension: $name"

cd $inputDir
find . -type d -exec mkdir -p "$outputDir"/{} \;


function meshlab_poisson_reconstruct () {
	iFile="$1"
	iName="$(basename $iFile)"
	# remove last extension
	iName="${iName%.*}"
	iDir="$(dirname $iFile)"
	oFile="$3/$iName".ply
    sFile="$2"
	# meshlab.meshlabserver -i $iFile -o $oFile -m vn -s $sFile
	# echo "meshlab.meshlabserver -i $iFile -o $oFile -s $sFile"
	if [ ! -f "$oFile" ]; then
		meshlab.meshlabserver -i $iFile -o $oFile -m vn -s $sFile
		# meshlabserver -i $oFile -o $oFile2 -s $sFile2
	fi
}
export -f meshlab_poisson_reconstruct

echo $scriptFile
find . -type f -wholename "$name"
find . -type f -wholename "$name" | xargs -P $num_procs -I % bash -c 'meshlab_poisson_reconstruct "$@"' _ % $scriptFile $outputDir
cd $myDir
