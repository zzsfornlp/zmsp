#

# prepare data for conll12
# simply following http://cemantix.org/data/ontonotes.html

# step 1: unpack LDC2013T19.tgz
...

# step 2: get conll files
wget https://github.com/ontonotes/conll-formatted-ontonotes-5.0/archive/v12.tar.gz
tar -zxvf v12.tar.gz
mv conll-formatted-ontonotes-5.0-12/* .
rmdir conll-formatted-ontonotes-5.0-12

# step 3: get script and convert to conll
# wget http://conll.cemantix.org/2012/download/conll-2012-scripts.v3.tar.gz
# tar -zxvf conll-2012-scripts.v3.tar.gz
wget http://ontonotes.cemantix.org/download/conll-formatted-ontonotes-5.0-scripts.tar.gz
tar -zxvf conll-formatted-ontonotes-5.0-scripts.tar.gz
# need to use python2
conda activate py27
bash conll-formatted-ontonotes-5.0/scripts/skeleton2conll.sh -D ontonotes-release-5.0/data/files/data/ conll-formatted-ontonotes-5.0/
conda deactivate

# --
# step 3.5: check files (need to be at "conll-formatted-ontonotes-5.0" dir)
cd conll-formatted-ontonotes-5.0
for wset in train development test conll-2012-test; do
wget http://ontonotes.cemantix.org/download/english-ontonotes-5.0-${wset}-document-ids.txt
cat english-ontonotes-5.0-${wset}-document-ids.txt | sort >ref-${wset}.txt
cd data/${wset}
find data -name "*.gold_conll" | sed "s/\.gold_conll//" | sort >${OLDPWD}/cur-${wset}.txt
cd ${OLDPWD}
diff -s ref-${wset}.txt cur-${wset}.txt
done
cd ..
# --

# step 4: finally concat them together
for wset in train development test conll-2012-test; do
  find conll-formatted-ontonotes-5.0/data/${wset} -name "*.gold_conll" -exec cat {} \; >${wset}.conll
done
ln -s development.conll dev.conll
