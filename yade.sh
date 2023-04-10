for i in $(seq 2401 2700); do
    cp ./Yade/simple_compression.py ./datanew/sample$i
    cd ./datanew/sample$i

    tmux new -d -s "Yade-"${i} 'yade simple_compression.py'
    cd ../..
done