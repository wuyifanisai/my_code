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







