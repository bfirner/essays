
set terminal pngcairo enhanced size 1400,540

set key outside center top horizontal

set key outside top
set key autotitle columnhead
set datafile separator ","

set xrange [0:1]
set yrange [0:1.5]

set output "figures/least-squares-no-noise.png"
`python3 poly-fit.py > tmp.dat`
plot "tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w points lw 1.5

set output "figures/least-squares-with-noise.png"
`python3 poly-fit-noise.py > tmp.dat`
plot "tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w points, for [j=4:5] '' u 1:j w lp lw 1.5

set output "figures/dnn-with-noise.png"
`python3 dnn-fit-noise.py > tmp.dat`
plot "tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w points, for [j=4:4] '' u 1:j w lp lw 1.5

set output "figures/dnn-with-noise-l2.png"
`python3 dnn-fit-l2.py > tmp.dat`
plot "tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w points, for [j=4:4] '' u 1:j w lp lw 1.5

set output "figures/dnn-with-noise-dropout.png"
`python3 dnn-fit-noise-dropout.py > tmp.dat`
plot "tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w points, for [j=4:4] '' u 1:j w lp lw 1.5
