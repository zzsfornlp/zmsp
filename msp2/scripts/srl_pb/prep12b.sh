#

# use the splittings from actual conll12 (but data from the conll13 paper)
# https://conll.cemantix.org/2012/data.html

# get data
wget http://conll.cemantix.org/2012/download/conll-2012-train.v4.tar.gz
wget http://conll.cemantix.org/2012/download/conll-2012-development.v4.tar.gz
wget http://conll.cemantix.org/2012/download/test/conll-2012-test-official.v9.tar.gz
for ff in *.gz; do tar -zxvf $ff; done

# get doc ids
find conll-2012/v4/data/train/data/english/ -name "*.v4_auto_skel" | sed "s/conll-2012\/v4\/data\/train\///;s/\.v4_auto_skel//" | sort >ids_train.txt
find conll-2012/v4/data/development/data/english/ -name "*.v4_auto_skel" | sed "s/conll-2012\/v4\/data\/development\///;s/\.v4_auto_skel//"  | sort >ids_dev.txt
find conll-2012/v9/data/test/data/english/ -name "*.v9_auto_skel" | sed "s/conll-2012\/v9\/data\/test\///;s/\.v9_auto_skel//" | sort >ids_test.txt

# concat all the files
data_dir=../conll12/conll-formatted-ontonotes-5.0/data/
ln -s ${data_dir}/train data_train
ln -s ${data_dir}/development data_dev
ln -s ${data_dir}/conll-2012-test data_test
for wset in train dev test; do
while IFS= read -r line; do
  cat data_${wset}/${line}.gold_conll;
done <ids_${wset}.txt >${wset}.conll
done
