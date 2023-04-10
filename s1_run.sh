for i in $(seq 121 180); do
    mkdir "./datanew/sample"${i}
    cp ./EDMD/spheres ./datanew/sample$i
    cd ./datanew/sample$i
    tmux new -d -s "compute-"${i} './spheres ../../s1_input'
    cd ../..
done


# for i in $(seq 801 801); do
# #    mkdir "sample"${i}
# #    cp spheres ./sample$i
#     cd sample$i
#     let num=i-800
#     tmux new -d -s "computenew-"${i} './spheres ./input'${num}
#     cd ..
# done

# for i in $(seq 1 100); do
#     let num=i+800
#     cp input1 ./sample$num/input$i
# done

# for i in $(seq 1 100); do
#     let num=i+800
#     cp input1 ./sample$num/input$i
# done


