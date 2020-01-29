direct=/mnt/nfs/work1/mccallum/smysore/material_science_framex/datasets_raw/msann_july2018-release
fname=$direct/sfex
for split in train test dev
do
	while read p; do
		cp -v $direct/data/$p.{txt,ann} $split/
	done <$fname-$split-fnames.txt
done
