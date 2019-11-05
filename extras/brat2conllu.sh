for fil in wlpdata/protocol_*.txt
do
	echo $(basename -- ${fil%.txt})
	fname=$(basename -- ${fil%.txt})
	perl brat2conllu.pl $fil ${fil%.txt}.ann > conll_wetlab/$fname.conllu
done
