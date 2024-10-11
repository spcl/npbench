#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Adding id line numbers corresponding to 1-st column entries to csv files"
    echo "Usage: $0 \"regex-for-csv-files\" "
    exit 1
fi
awk '{print $1}' $1 | sort | uniq | nl > efurdevkicnuecugkeckdctcrfijlvkv
mkdir -p sortedCSV
for FILE in $(ls $1); do
  cp $FILE ectfgttdencenbuchtldlvvhdhrbrreg
  while IFS= read -r line; do
    BM=$(echo $line |  awk '{print $2}')
    sed -i "s/$BM/$line/g"  tmp
  done < efurdevkicnuecugkeckdctcrfijlvkv
  awk '{$1=$1;print}' ectfgttdencenbuchtldlvvhdhrbrreg > sortedCSV/$FILE
done
rm -rf efurdevkicnuecugkeckdctcrfijlvkv
