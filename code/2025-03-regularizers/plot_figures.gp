
set terminal pngcairo enhanced size 1400,540 font ",24"

set key outside center top horizontal

set key outside top
set key autotitle columnhead
set datafile separator ","

set xrange [0:1]
set yrange [0:1.5]

set output "figures/least-squares-no-noise.png"
`python3 poly-fit.py > tmp.dat`
plot "tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w points ps 2 lw 1.5

set output "figures/least-squares-with-noise.png"
`python3 poly-fit-noise.py > tmp.dat`
plot "tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w points ps 2, for [j=4:5] '' u 1:j w lp lw 1.5

set output "figures/dnn-with-noise.png"
`python3 dnn-fit-noise.py > tmp.dat`
plot "tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w points ps 2, for [j=4:4] '' u 1:j w lp lw 1.5

set output "figures/dnn-with-noise-l2.png"
`python3 dnn-fit-l2.py > tmp.dat`
plot "tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w points ps 2, for [j=4:4] '' u 1:j w lp lw 1.5

set output "figures/dnn-with-noise-dropout.png"
`python3 dnn-fit-noise-dropout.py > tmp.dat`
plot "tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w points ps 2, for [j=4:4] '' u 1:j w lp lw 1.5

set output "figures/dnn-manual-fit.png"
`python3 dnn-manual-fit.py > tmp.dat`
plot "<grep ',' tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w lp lw 1.5

set output "figures/dnn-manual-fit-errors.png"
`python3 dnn-manual-fit-errors.py > tmp.dat`
plot "<grep ',' tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w lp lw 1.5

set output "figures/dnn-manual-fit-errors-l2.png"
`python3 dnn-manual-fit-errors-l2.py > tmp2.dat`
plot "<grep ',' tmp2.dat" u 1:2 w l lw 1.5, '' u 1:3 w lp lw 1.5

set xrange [-1:2]
set yrange [-1:1.5]
set output "figures/dnn-with-noise-outrange.png"
`python3 dnn-fit-noise-outrange.py > tmp.dat`
plot "tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w points ps 2, for [j=4:4] '' u 1:j w lp lw 1.5

set output "figures/dnn-with-noise-outrange-l2.png"
`python3 dnn-fit-noise-outrange-l2.py > tmp.dat`
plot "tmp.dat" u 1:2 w l lw 1.5, '' u 1:3 w points ps 2, for [j=4:4] '' u 1:j w lp lw 1.5

set xrange [0:1]
set yrange [0:1.5]

set xrange [-6.5:6.5]
set yrange [-6.5:6.5]
set ylabel "Output"
set xlabel "Input"
set title "ReLU"
relu(x)=x<0?0:x
set output "figures/relu.png"
plot relu(x) w l notitle

set xrange [-1:1]
set key noautotitle
set ylabel "Output"
set xlabel "x"
set output "figures/toy_example2.png"
plot "toy_example2.dat" u 1:2 w lp ps 2 lw 1.5
