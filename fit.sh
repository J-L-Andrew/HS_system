for i in $(seq 1 40); do
    cp ./Yade/analysis ./data/sample$i
    cd ./data/sample$i

    tmux new -d -s "Fit-"${i} './analysis'
    cd ../..
done