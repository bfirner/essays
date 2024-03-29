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

# First build the index and sidebar
rm index.md
rm sidebar.md

before_html="-B common/style.html -B common/pre_content_sidebar.html"
after_html="-A sidebar.html -A common/post_content_sidebar.html -A common/footer.html"

for md in *.md;
do
    touch index.md
    touch sidebar.md
    title=`head -n 1 ${md}`
    basename=${md%%.md}
    date=${md%%-*}
    year=$(echo $date | cut -c 1-4)
    month=$(echo $date | cut -c 5-6)
    day=$(echo $date | cut -c 7-8)
    # Make a shorter name for the sidebar
    if [[ ${#title} > 40 ]]; then
        shorttitle=$(echo $title | cut -c1-37)
        shorttitle=$(echo $shorttitle "...")
    else
        shorttitle=$title
    fi
    # Put the new item at the top of the index and sidebar files. If the files are named with their
    # dates first then this should put the newest ones on top.
    # The sponge command is from the moreutils package.
    (echo -e "$year-$month-$day: [$title](${basename}.html)\n"; cat index.md) | sponge index.md
    (echo -e ["$year-$month: $shorttitle](${basename}.html)\n"; cat sidebar.md) | sponge sidebar.md
done

# Make the sidebar a div that follows main.
(echo -e '</main><div class="sidebar">\n'; cat sidebar.md) | sponge sidebar.md
echo '</div>' >> sidebar.md

# Create the index html.
pandoc -o html/index.html index.md --strip-comments --standalone \
    --mathml \
    -B common/style.html -A common/footer.html --metadata pagetitle="Index"
# Create the sidebar
pandoc -o sidebar.html sidebar.md --strip-comments --mathml

# Create the individual pages
for md in $(ls *.md | grep -vP "(index.md|sidebar.md)");
do
    touch index.md
    title=`head -n 1 ${md}`
    basename=${md%%.md}
    date=${md%%-*}
    year=$(echo $date | cut -c 1-4)
    month=$(echo $date | cut -c 5-6)
    day=$(echo $date | cut -c 7-8)
    echo "Writing \"$title\" to html/${basename}.html"
    # Now output the html
    pandoc -o html/${basename}.html ${md} --strip-comments --standalone \
        --mathml \
        $before_html $after_html --metadata pagetitle="${title}"
done
