#

# example of preparing parallel corpus and run with fast_align
#wget http://statmt.org/europarl/v7/europarl.tgz
mkdir -p europarlv7; cd europarlv7
for cl in de fr it es pt 'fi'; do
  wget https://www.statmt.org/europarl/v7/${cl}-en.tgz
  tar -zxvf ${cl}-en.tgz
  bash prep.sh en ${cl} europarl-v7.${cl}-en
#  OUT_DIR=_tmp_${cl} bash align.sh europarl-v7.${cl}-en.en-${cl}.clean.fastalign /dev/null /dev/null
done
# wc *.fastalign # =>
#   1901781  104760285  627625564 europarl-v7.de-en.en-de.clean.fastalign
#   1950655  111714601  627878935 europarl-v7.es-en.en-es.clean.fastalign
#   1912941   91898631  601503429 europarl-v7.fi-en.en-fi.clean.fastalign
#   1990884  117547171  674313638 europarl-v7.fr-en.en-fr.clean.fastalign
#   1892649  110388781  642045740 europarl-v7.it-en.en-it.clean.fastalign
#   1947070  112086751  635294230 europarl-v7.pt-en.en-pt.clean.fastalign
#  11595980  648396220 3808661536 total
