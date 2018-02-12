#!/bin/bash
clear
echo '**************simple math in shell**************'
a=10
b=3

d=$((a+b))
e=$((a-b))
f=$((a*b))
g=$((a/b))

echo "a="$a
echo $b
echo $c
echo ${d}
echo ${e}
echo ${f}
echo ${g}
echo ''

echo "************simple string dealing in shell ****************"
echo 'the string is:'
lines="hello world"
echo $lines
echo 'the length of this string is:'
echo ${#lines}
#echo 'substring of this string is:'
#echo ${lines:1:4}

echo 'read input..'
echo 'your name is :'
read name
echo 'hello '${name}
echo ''

echo '**************** assmbles **************'
echo 'a'||'b'
pwd ; ls
ls && pwd
echo ''

echo '************* pipeline ***********'

#!/bin/bash
echo 'this script is s.sh'
list="a b c d" 
for alpha in $list;do      
echo $alpha 
done

cd shell 
pwd 
ls
mkdir shell_tut
cd shell_tut

for i in $list;
do
touch test_$i.txt
done

for Variable in {1..3};
do     
echo "$Variable" 
done










