#! /usr/bin/bash

# This script will use pandoc to render all of the markdown files in the this folder into html.
# Then an index.md file will be created and also converted to html.
# All outputs will be placed in the html folder.

mkdir -p html

# Descend into the figures directory, build the figures, and copy them into the html directory.
mkdir -p html/figures
cd figures
for I in *.gp; do
    gnuplot $I;
done
cp *.png ../html/figures/
cd ../

rm index.md
for md in *.md;
do
    touch index.md
    title=`head -n 1 ${md}`
    basename=${md%%.md}
    date=${md%%-*}
    year=$(echo $date | cut -c 1-4)
    month=$(echo $date | cut -c 5-6)
    day=$(echo $date | cut -c 7-8)
    echo "Writing \"$title\" to html/${basename}.html"
    # Put the new item at the top of the index file. If the files are named with their dates first
    # then this should put the newest ones on top.
    # The sponge command is from the moreutils package.
    (echo -e "$year-$month-$day: [$title](${basename}.html)\n"; cat index.md) | sponge index.md
    # Now output the html
    pandoc -o html/${basename}.html ${md} --strip-comments --standalone \
        --mathml \
        -B common/style.html -A common/footer.html --metadata pagetitle="${title}"
done

# Finally create the index html.
pandoc -o html/index.html index.md --strip-comments --standalone \
    --mathml \
    -B common/style.html -A common/footer.html --metadata pagetitle="Index"
