set terminal postscript eps color size 30,25 solid enhanced font 'TimesNewRoman' 190 linewidth 16

if (!exists("the_output_file")) the_output_file='eff_dim.eps'
if (!exists("the_title")) the_title=''
set key inside right top
set style line 1 linecolor  rgb "#0072bd" linetype 1 linewidth 2 pointtype 1 pointsize 10
set style line 2 linecolor  rgb '#d95319' linetype 1 linewidth 2 pointtype 7 pointsize 10
set style line 3 linecolor  rgb '#edb120' linetype 1 linewidth 2 pointtype 3 pointsize 10
set style line 4 linecolor  rgb '#7e2f8e' linetype 1 linewidth 2 pointtype 9 pointsize 10
set style line 5 linecolor  rgb '#77ac30' linetype 1 linewidth 2 pointtype 14 pointsize 10
set style line 6 linecolor  rgb '#4dbeee' linetype 1 linewidth 2 pointtype 12 pointsize 10

set xtics nomirror
set tics scale 8
set xtics 0.2
set ytics 0.1
set ylabel "Normalized effective dimension"

set output the_output_file
set title the_title


stats "Effect_dime_x.txt" using 1 name 'x' nooutput
set xlabel sprintf("Number of data (x10^%.0f)",log10(x_max))

plot [][0:1] "<paste Effect_dime_x.txt Effect_dime.txt" using ($1/x_max):2 title "" ls  3 with linespoints


