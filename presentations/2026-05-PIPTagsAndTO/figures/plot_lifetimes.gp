#set terminal png nocrop enhanced truecolor font '/usr/share/fonts/TTF/LiberationSans-Bold.ttf'
set terminal pngcairo size 1920,1080 font "CourierPrime-Bold" fontscale 3 enhanced

set style line 1 lc rgbcolor "0x302E2585" pt 5 ps 3 lw 3
set style line 2 lc rgbcolor "0x30337538" pt 7 ps 3 lw 3
set style line 3 lc rgbcolor "0x305DA899" pt 9 ps 3 lw 3
set style line 4 lc rgbcolor "0x3094CBEC" pt 11 ps 3 lw 3
set style line 5 lc rgbcolor "0x30AB94EC" pt 13 ps 3 lw 3

set ylabel "Maximum Possible\nRadio Duty Cycle (%)"

set xlabel "Desired Lifetime (years)"

#Sleep current is 400 nano Amps for CC2500
cc_sleep = 0.0004
#radio costs
cc_tx = 21.2
cc_rx = 18.3

#For the CC2550 transmitter
#Sleep current is 200 nano Amps
txo_sleep = 0.0002
#radio transmission is slightly better
txo_tx = 19.4

#12MHz @ 3V, MSP430G2x53
msp_on = 3.0
#Sleep current is 0.1 micro Amps
msp_sleep = 0.0001

#Timing is in milliseconds
delta_cs = 0.200
delta_ack = 0.250
c_tx = 24.2
c_rx = 21.3

#Timing is in milliseconds
delta_req = 2.0*0.176
delta_resp = 2.0*0.352
delta_turn = 0.150
delta_proto = 2.0*0.560
delta_idle = 0.900
ble_radio = 19.0
ble_low = 0.0005
ble_idle = 6.7

pclear(delta_tx, interval, num) = (1.0 - (delta_cs + delta_tx + delta_ack)/interval)**(num-1.0)

cost_zig(delta_tx, interval, num) = (ecs = 1.0/pclear(delta_tx, interval, num),\
  (ecs*delta_cs + delta_ack)*c_rx + delta_tx*c_tx +\
  (interval - ecs*delta_cs - delta_tx - delta_ack)*(msp_sleep+cc_sleep))

psucc(interval, num) = (1.0 - (2.0*delta_req + delta_turn + delta_resp)/interval)**(num - 1.0)

cost_ble(delta_tx, interval, num) = (ereq = 1.0/psucc(interval, num),\
  ereq*(delta_turn*ble_idle + (delta_req + delta_resp)*ble_radio) +\
  (delta_proto + delta_tx)*ble_radio + delta_idle*ble_idle +\
  (interval - ereq*(delta_turn+delta_req+delta_resp) - delta_proto - delta_tx - delta_idle)*ble_low )

cost_to(delta_tx, interval) = (interval - delta_tx)*(msp_sleep+cc_sleep) + delta_tx*(msp_on+cc_tx)

#cost_tm(delta_tx, interval, num) = cost_to(delta_tx, interval) + (delta_tx*c_tx+delta_ack*c_rx)*(1.0/0.99)**2.0
e_tm_repeat(num, ack_period) = 1.0 / (1.0 - (2.0*delta_ack / ack_period))**(num - 1.0)
cost_tm(delta_tx, interval, num, ack_interval) = (perc_regular = (ack_interval - 1.0)/ack_interval,\
   (perc_regular)*cost_to(delta_tx, interval) +\
   (1.0 - perc_regular)*((delta_tx*c_tx+delta_ack*c_rx)*e_tm_repeat(num, ack_interval*interval) + (msp_sleep+cc_sleep)*(interval - (delta_tx+delta_ack)*e_tm_repeat(num, ack_interval*interval))) )

radio_current = 10
sleep_current = 0.001
energy_left(years, energy) = (energy - (years * sleep_current * 24 * 365))
duty_cycle(years, energy) = 100*(energy_left(years, energy) / radio_current) / (365*24*years)

mcu_energy_left(years, energy) = (energy - (years * (sleep_current+mcu_sleep) * 24 * 365))
mcu_duty_cycle(years, energy) = 100*(mcu_energy_left(years, energy) / (radio_current + mcu_current)) / (365*24*years)

set output "maximum_duty_cycle.png"
set xrange [1:20]
set logscale y
set yrange [0.001:16]

set label "225 mAh\n(CR2032)" at 1.8*2.25, 1.1*duty_cycle(2*2.25, 225) center front
#set label "610 mAh\n(CR2450)" at 2*4, 1.1*duty_cycle(2*4, 610) center front
set label "1200 mAh\n(AAA alkaline)" at 1.6*6, 1.1*duty_cycle(2*6, 1200) center front
set label "2700 mAh\n(AA alkaline)" at 1.8*9, 1.1*duty_cycle(2*9, 2700) center front
#set label "8000 mAh\n(C alkaline)" at 2*4, 1.1*duty_cycle(2*4, 8000) center front
set label "12000 mAh\n(D alkaline)" at 2*6.75, 1.1*duty_cycle(2*6.75, 12000) center front

plot duty_cycle(x, 225) w l ls 1 lw 4 not,\
     duty_cycle(x, 1200) w l ls 2 lw 4 not,\
     duty_cycle(x, 2700) w l ls 3 lw 4 not,\
     duty_cycle(x, 12000) w l ls 4 lw 4 not
#     duty_cycle(x, 610) w l ls 2 lw 4 not,\
#     duty_cycle(x, 8000) w l ls 5 lw 4 not,\


unset label

set key top left
#set grid ytics

unset yrange
set output "MAC_lifetimes.eps"
set xrange [0.1:60]

unset logscale y
set logscale x

#Need to convert energy consumptions to rates to find lifetimes
#en_per_sec(on_per_second) = (on_per_second*(msp_on + radio_current)+(1-on_per_second)*(cc_sleep + mcu_sleep))
lifetime(en_per_sec, total_energy) = (total_energy/en_per_sec)/(365*24)

ble_lifetime(interval, pack_duration) = lifetime(cost_ble(pack_duration, interval, 1.0)/interval, 220)
zig_lifetime(interval, pack_duration) = lifetime(cost_zig(pack_duration, interval, 1.0)/interval, 220)
to_lifetime(interval, pack_duration) = lifetime(cost_to(pack_duration, interval)/interval, 220)
tm_lifetime(interval, pack_duration, ack_interval) = lifetime(cost_tm(pack_duration, interval, 1.0, 10.0)/interval, 220)

ble_lifetime_coll(interval, pack_duration, num) = lifetime(cost_ble(pack_duration, interval, num)/interval, 220)
zig_lifetime_coll(interval, pack_duration, num) = lifetime(cost_zig(pack_duration, interval, num)/interval, 220)
tm_lifetime_coll(interval, pack_duration, num, ack_interval) = lifetime(cost_tm(pack_duration, interval, num, 10.0)/interval, 220)

set ylabel "Sensor Lifetime (years)"
set xlabel "Desired Packet Interval (seconds)"

#set label "Threshold for 1 year with a coin cell battery" at 59, 0.01+duty_cycle(1, 225) right
#set label "... 5 years with a coin cell battery" at 59, 0.01+duty_cycle(5, 225) right
#set label "... 10 years with a coin cell battery" at 59, 0.01+duty_cycle(10, 225) right

#plot mcu_duty_cycle(1, 225) lt 1 notitle, mcu_duty_cycle(5, 225) lt 1 notitle, mcu_duty_cycle(10, 225) lt 1 notitle

set yrange [*:*]

plot ble_lifetime(1000.0*x, 1.0) lt 3 title "Bluetooth Low Energy",\
     zig_lifetime(1000.0*x, 1.0) lt 4 title "CSMA (802.15.4 transmitter to powered receiver)",\
     to_lifetime(1000.0*x, 1.0) lt 1 title "Transmit Only",\
     tm_lifetime(1000.0*x, 1.0, 10.0) lt 2 title "Transmit Mostly (request ACK every 10 packets)"

set xlabel "Number of Transmitters"
unset logscale x
set xrange [1:500]
unset logscale y
#set yrange[0.01:10]
set yrange [0:1.2]

set key bottom left

#Plot the channel utilization by different MAC protocols with different contention levels
set output "MAC_lifetime_contention.eps"

plot ble_lifetime_coll(1000.0, 1.0, x) lt 3 title "Bluetooth Low Energy",\
     zig_lifetime_coll(1000.0, 1.0, x) lt 4 title "CSMA (802.15.4 transmitter to powered receiver)",\
     to_lifetime(1000.0, 1.0) lt 1 title "Transmit Only",\
     tm_lifetime_coll(1000.0, 1.0, x, 10.0) lt 2 title "Transmit Mostly (request ACK every 10 packets)"

#This is a boring graph that only shows the relative costs of the radio
set output "MAC_lifetime_packet_size.eps"

set key top right
set yrange [*:*]
set xrange [0.2:10]
set logscale y
set xlabel "Data Packet Duration (milliseconds)"

plot ble_lifetime_coll(1000.0, x, 100) lt 3 title "Bluetooth Low Energy",\
     zig_lifetime_coll(1000.0, x, 100) lt 4 title "CSMA (802.15.4 transmitter to powered receiver)",\
     to_lifetime(1000.0, x) lt 1 title "Transmit Only",\
     tm_lifetime_coll(1000.0, x, 100, 10.0) lt 2 title "Transmit Mostly (request ACK every 10 packets)"

