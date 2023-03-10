for i in $(seq 1 1); do
    sed -i '10c xl='${i} modulus_used_sphere.py
    gnome-terminal -t "xl="${i} -- yade modulus_used_sphere.py
done





