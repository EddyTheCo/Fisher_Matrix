set terminal postscript eps color size 25,25 solid enhanced font 'TimesNewRoman' 190 linewidth 16
if (!exists("the_output_file")) the_output_file='eig_dist.eps'
if (!exists("the_title")) the_title=''
set style line 1 linecolor  rgb "#0072bd" linetype 1 linewidth 2 pointtype 1 pointsize 10
set style line 2 linecolor  rgb '#d95319' linetype 1 linewidth 2 pointtype 7 pointsize 10
set style line 3 linecolor  rgb '#edb120' linetype 1 linewidth 2 pointtype 3 pointsize 10
set style line 4 linecolor  rgb '#7e2f8e' linetype 1 linewidth 2 pointtype 9 pointsize 10
set style line 5 linecolor  rgb '#77ac30' linetype 1 linewidth 2 pointtype 14 pointsize 10
set style line 6 linecolor  rgb '#4dbeee' linetype 1 linewidth 2 pointtype 12 pointsize 10

set style fill solid 5 border lc black
set style boxplot outliers candlesticks pointtype 4
set style boxplot medianlinewidth 2.0
set errorbars large
set format y "%.1f";

set style data boxplot
set boxwidth  2.5
set pointsize 5
unset key
set tics nomirror
set tics scale 8
set ylabel "Eigenvalues"


set output the_output_file
set xtics (the_title 1) scale 0.0

plot [][] sprintf("<awk '1' RS='[[:space:]]+' Spectrum.txt") using (1):1 ls 5

