#!/bin/bash

if [ "$3" == "" ] ; then 
 echo "Usage: <original bvecs> <rotated bvecs> <ecclog>"
 echo ""
 echo "<ecclog> is the output log file from ecc"
 echo ""
 exit 1;
fi

i=$1
o=$2
ecclog=$3

if [ ! -e $1 ] ; then
    echo "Source bvecs $1 does not exist!"
    exit 1
fi
if [ ! -e $ecclog ]; then
    echo "Ecc log file $3 does not exist!"
    exit 1
fi

ii=1
rm -f $o
tmpo=${o}$$

cat ${ecclog} | while read line; do
    if [ "$line" == "" ];then break;fi
    read line;
    read line;
    read line;

    echo $line  > $tmpo
    read line    
    echo $line >> $tmpo
    read line    
    echo $line >> $tmpo
    read line    
    echo $line >> $tmpo
    read line   

    ####print lines of $tmpo | grab nth line | print nth value of line
    m11=`cat $tmpo | sed '1q;d' | awk '{print $1}'`
    m12=`cat $tmpo | sed '1q;d' | awk '{print $2}'`
    m13=`cat $tmpo | sed '1q;d' | awk '{print $3}'`
    m21=`cat $tmpo | sed '2q;d' | awk '{print $1}'`
    m22=`cat $tmpo | sed '2q;d' | awk '{print $2}'`
    m23=`cat $tmpo | sed '2q;d' | awk '{print $3}'`
    m31=`cat $tmpo | sed '3q;d' | awk '{print $1}'`
    m32=`cat $tmpo | sed '3q;d' | awk '{print $2}'`
    m33=`cat $tmpo | sed '3q;d' | awk '{print $3}'`

    X=`cat $i | head -n $ii| tail -n 1 | awk -v N=1 '{print $N}' | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}'`
    Y=`cat $i | head -n $ii| tail -n 1 |awk -v N=2 '{print $N}' | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}'`
    Z=`cat $i | head -n $ii| tail -n 1 | awk -v N=3 '{print $N}' | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}'`

    rX=`echo "scale=7;  ($m11 * $X) + ($m12 * $Y) + ($m13 * $Z)" | bc -l`
    rY=`echo "scale=7;  ($m21 * $X) + ($m22 * $Y) + ($m23 * $Z)" | bc -l`
    rZ=`echo "scale=7;  ($m31 * $X) + ($m32 * $Y) + ($m33 * $Z)" | bc -l`
    if [ "$ii" -eq 1 ];then
    echo $rX > $o;echo $rY >> $o;echo $rZ >> $o
    else
    cp $o $tmpo
    (echo $rX;echo $rY;echo $rZ) | paste $tmpo - > $o
    fi
    
    let "ii+=1"

done
rm -f $tmpo
