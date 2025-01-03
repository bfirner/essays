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
