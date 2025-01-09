
set terminal pngcairo enhanced

set key outside center top horizontal

set output "../figures/2025-01-function-from-basis-functions.png"
basis(x, offset)=2**(-(x - offset)**2)
plot for [j=-2:2] basis(x, j) w l title "basis ".(j+2), basis(x, -2)+basis(x, -1)+basis(x, 0)+basis(x, 1)+basis(x, 2) w l title "Target function"

set key inside top
set key autotitle columnhead
set datafile separator ","
set yrange [0:3]

set xrange [-2.5:2.5]

#set output "../figures/2025-01-function-from-x.png"
#`python3 2025-01-building-piecewise-fit-step-w-normalizers.py --test 1 | grep '^bigmodel1000' > tmp.dat`
#plot "tmp.dat" u 2:3 w l lw 1.5, '' u 2:4 w points lw 1.5

#set output "../figures/2025-01-function-from-x-better.png"
#`python3 2025-01-building-piecewise-fit-step-w-normalizers.py --test 2 | grep '^bigmodel1000' > tmp.dat`
#plot "tmp.dat" u 2:3 w l lw 1.5, '' u 2:4 w points lw 1.5

set key inside bottom

#set output "../figures/2025-01-function-from-x-human.png"
#`python3 2025-01-building-piecewise-fit-step-w-normalizers.py --test 3 | grep '^humanmodel' > tmp.dat`
#plot "tmp.dat" u 2:3 w l lw 1.5, '' u 2:4 w points lw 1.5

#set output "2025-01-function-from-x-human-components.png"
#set yrange [*:*]
#plot "tmp.dat" u 2:3 w l lw 1.5, '' u 2:4 w points lw 1.5, for [j=5:13] '' u 2:j w lines dt 2

set yrange [0:3]
set output "../figures/2025-01-function-from-x-human-bad.png"
`python3 2025-01-building-piecewise-fit-step-w-normalizers.py --test 4 | grep '^humanmodel_bad' > tmp.dat`
plot "tmp.dat" u 2:3 w l lw 1.5, '' u 2:4 w points lw 1.5

#set output "train-linear-curve-with-gap-sum-functions.png"
#plot "<grep 'Step1' regularizers-example-results.dat" u 2:7 ps 2, '' u 2:8 ps 2, "<grep 'Step2' regularizers-example-results.dat" u 2:3 w lp, '' u 2:4 w lp
#
#set output "train-linear-curve-with-gap-linear-function.png"
#plot "<grep 'Step3' regularizers-example-results.dat" u 2:3 ps 2, '' u 2:4 ps 2, "<grep 'Step4' regularizers-example-results.dat" u 2:3 w lp, '' u 2:4 w lp
#
#set output "train-linear-curve-with-gap-linear-function-small-capacity.png"
#plot "<grep 'Step9' regularizers-example-results.dat" u 2:3 ps 2, '' u 2:4 ps 2, "<grep 'Step10' regularizers-example-results.dat" u 2:3 w lp, '' u 2:4 w lp
#
#set key outside center top horizontal maxcols 3 maxrows 3 spacing 1 width -5
#set grid
#
#set yrange [*:*]
#set output "train-linear-curve-small-capacity-human-pieces.png"
#plot "<grep Step1 train-linear-piecewise-results.dat" u 2:3 w lp lw 1 t "Target", "<grep Step1 train-linear-piecewise-results.dat" u 2:4 w lp t "Preset Output", "<grep Step3 train-linear-piecewise-results.dat" u 2:3 w lp dt 2, '' u 2:4 w lp dt 2, '' u 2:5 w lp dt 2, '' u 2:6 w lp dt 2, '' u 2:7 w lp dt 2, '' u 2:8 w lp dt 2, '' u 2:9 w lp dt 2, '' u 2:10 w lp dt 2, '' u 2:11 w lp dt 2
#
#
#set output "train-linear-curve-small-capacity-trained-pieces.png"
#plot "<grep Step1 train-linear-piecewise-results.dat" u 2:3 w lp lw 1 t "Target", "<grep Step2 train-linear-piecewise-results.dat" u 2:4 w lp t "Trained Output", "<grep Step4 train-linear-piecewise-results.dat" u 2:3 w lp dt 2, '' u 2:4 w lp dt 2, '' u 2:5 w lp dt 2, '' u 2:6 w lp dt 2, '' u 2:7 w lp dt 2, '' u 2:8 w lp dt 2, '' u 2:9 w lp dt 2, '' u 2:10 w lp dt 2, '' u 2:11 w lp dt 2
#
## We would rather use a webp over a gif, but we can't because Ubuntu is years behind.
##set terminal webp animate delay 50
#
#set key outside center top horizontal maxcols 3 maxrows 3 spacing 1 width 0
#
##set terminal gif animate delay 2 loop 0
##set output "train-linear-curve-small-capacity-trained-pieces-animation.gif"
##set yrange [0:4.1]
##do for [step=0:1000:1] {
##    set title "Step ".step
##    plot "<grep '^Step".step.",' train-linear-piecewise-steps.dat" u 2:3 w lp lw 1.5, '' u 2:4 w lp lw 1.5, for [j=5:13] '' u 2:j w lp dt 2
##}
#unset output
#
##set terminal gif animate delay 2 loop 0
##set output "train-linear-curve-small-capacity-trained-pieces-leakyrelu-animation.gif"
##set yrange [0:4.1]
##do for [step=0:1000:1] {
##    set title "Step ".step
##    plot "<grep '^leaky".step.",' train-linear-piecewise-steps.dat" u 2:3 w lp lw 1.5, '' u 2:4 w lp lw 1.5, for [j=5:13] '' u 2:j w lp dt 2
##}
##unset output
#
##set terminal gif animate delay 2 loop 0
##set output "train-linear-curve-small-capacity-trained-pieces-norelu-animation.gif"
##set yrange [0:4.1]
##do for [step=0:1000:1] {
##    set title "Step ".step
##    plot "<grep '^norelu".step.",' train-linear-piecewise-steps.dat" u 2:3 w lp lw 1.5, '' u 2:4 w lp lw 1.5, for [j=5:13] '' u 2:j w lp dt 2
##}
##unset output
#
#set terminal gif animate delay 2 loop 0
#set output "train-linear-curve-small-capacity-trained-pieces-bigger-animation.gif"
#set yrange [0:4.1]
#do for [step=0:1000:1] {
#    set title "Step ".step
#    plot "<grep '^Bigger".step.",' train-linear-piecewise-steps.dat" u 2:3 w lp lw 1.5, '' u 2:4 w lp lw 1.5, for [j=5:21] '' u 2:j w lp dt 2
#}
#unset output
#
#
##datasrc="train-linear-piecewise-steps-regularizers.dat"
##set terminal gif animate delay 2 loop 0
##set output "train-linear-curve-small-capacity-trained-pieces-bigger-animation.gif"
##set yrange [0:4.1]
##do for [step=0:1000:1] {
##    set title "Step ".step
##    plot "<grep '^Bigger".step.",' train-linear-piecewise-steps.dat" u 2:3 w lp lw 1.5, '' u 2:4 w lp lw 1.5, for [j=5:21] '' u 2:j w lp dt 2
##}
##unset output
#
