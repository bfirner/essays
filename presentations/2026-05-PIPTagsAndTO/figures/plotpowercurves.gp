set terminal pngcairo size 1920,1080 font "CourierPrime-Bold" fontscale 3 enhanced

set output 'powercurves.png'

unset border

set ylabel "Signal Power"
set xlabel "Receiver Location"
#set ytics ("Full Power" 1)
unset ytics
unset xtics
#set xtics axis ("Transmitter A" 0, "Transmitter B" 10)
set label at 0,2 center "Transmitter A's\nLocation" front
set label at 10,2 center "Transmitter B's\nLocation" front
set arrow from 0,1 to 0,0 front
set arrow from 10,1 to 10,0 front

set arrow heads from -7,-1.2 to 17,-1.2 front

#set label at 5,2 center "A's packet captured\nduring collision"
#set label at 2.5,2 center "Complete packet loss\nduring collision"
#set label at 4.5,2 center "B's packet captured\nduring collision"
#set arrow from 0.5,1.7 to 0.5,0
#set arrow from 2.5,1.7 to 2.5,0
#set arrow from 4.5,1.7 to 4.5,0

set yrange [0:10]
set xrange [-7:17]

#Image width to height ratio is 3:4

set key outside center top

factor = 16

set style fill pattern 2

t1(x) = 0.5**(abs(x)) 
t2(x) = factor*0.5**(abs(x-10)) > t1(x) ? 0.5**(abs(x-10)) : 0/0
coll(x) = t1(x) > factor*t2(x) ? 0/0 : t2(x) > factor*t1(x) ? 0/0 : t1(x) > t2(x) ? t1(x) : t2(x)
plot 10*t1(x) w filledcurves y1=0 title "Area where A's packet might be captured",\
     10*t2(x) w filledcurves y1=0 title "Area where B's packet might be captured",\
     10*coll(x) w filledcurves y1=0 title "Area where neither packet is received"
     #'antenna.png' binary filetype=png dx=0.03 dy=0.08 center=(0, 1.5) with rgbalpha notitle,\
     #'antenna.png' binary filetype=png dx=0.03 dy=0.08 center=(10, 1.5) with rgbalpha notitle
