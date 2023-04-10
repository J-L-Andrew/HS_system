for i in $(seq 2201 2400); do
    cd ./data/sample$i
    tmux new -d -s "compute-"${i} './spheres ../../s2_input'
    cd ../..
done