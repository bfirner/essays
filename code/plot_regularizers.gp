set datafile separator ","
set yrange [0:4.1]

set terminal pngcairo enhanced

set key bottom
set key autotitle columnhead

set output "train-linear-curve-with-gap-sum-functions.png"
plot "<grep 'Step1' regularizers-example-results.dat" u 2:7 ps 2, '' u 2:8 ps 2, "<grep 'Step2' regularizers-example-results.dat" u 2:3 w lp, '' u 2:4 w lp

set output "train-linear-curve-with-gap-linear-function.png"
plot "<grep 'Step3' regularizers-example-results.dat" u 2:3 ps 2, '' u 2:4 ps 2, "<grep 'Step4' regularizers-example-results.dat" u 2:3 w lp, '' u 2:4 w lp

set output "train-linear-curve-with-gap-linear-function-small-capacity.png"
plot "<grep 'Step9' regularizers-example-results.dat" u 2:3 ps 2, '' u 2:4 ps 2, "<grep 'Step10' regularizers-example-results.dat" u 2:3 w lp, '' u 2:4 w lp

set key outside center top horizontal maxcols 3 maxrows 3 spacing 1 width -5
set grid

set yrange [*:*]
set output "train-linear-curve-small-capacity-human-pieces.png"
plot "<grep Step1 train-linear-piecewise-results.dat" u 2:3 w lp lw 1 t "Target", "<grep Step1 train-linear-piecewise-results.dat" u 2:4 w lp t "Preset Output", "<grep Step3 train-linear-piecewise-results.dat" u 2:3 w lp dt 2, '' u 2:4 w lp dt 2, '' u 2:5 w lp dt 2, '' u 2:6 w lp dt 2, '' u 2:7 w lp dt 2, '' u 2:8 w lp dt 2, '' u 2:9 w lp dt 2, '' u 2:10 w lp dt 2, '' u 2:11 w lp dt 2


set output "train-linear-curve-small-capacity-trained-pieces.png"
plot "<grep Step1 train-linear-piecewise-results.dat" u 2:3 w lp lw 1 t "Target", "<grep Step2 train-linear-piecewise-results.dat" u 2:4 w lp t "Trained Output", "<grep Step4 train-linear-piecewise-results.dat" u 2:3 w lp dt 2, '' u 2:4 w lp dt 2, '' u 2:5 w lp dt 2, '' u 2:6 w lp dt 2, '' u 2:7 w lp dt 2, '' u 2:8 w lp dt 2, '' u 2:9 w lp dt 2, '' u 2:10 w lp dt 2, '' u 2:11 w lp dt 2
