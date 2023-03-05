mkdir -p html
for md in *.md;
do
    title=`head -n 1 ${md}`
    basename=${md%%.md}
    echo "Writing \"$title\" to html/${basename}.html"
    pandoc -o html/${basename}.html ${md} --strip-comments --standalone \
        --mathml \
        -B common/style.html -A common/footer.html --metadata pagetitle="${title}"
done
