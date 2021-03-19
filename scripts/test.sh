#!/bin/bash
date
for i in `seq 1 6`
do
{
 echo "sleep ${i}"
 sleep 2
}&
done
wait #等待执行完成
date