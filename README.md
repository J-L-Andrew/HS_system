# The mechanical properties of monodisperse hard-sphere system

### Simulation details
Hard-sphere packing generation with [the Lubachevsky–Stillinger (LS) algorithm](https://cims.nyu.edu/~donev/Packing/C++/index.html).
We set N=500, and change the expansion rate of particles to obtain many packing samples.
每个压缩率下算300个样本
可以参照https://pubs.rsc.org/en/content/articlepdf/2014/sm/c3sm52959b来写

面临的问题：0.7以上少，考虑引入defect？ 

另：对rattler问题要有交代，至少是collectively jammed。

Although these packings are obtained under different conditions, structural analysis suggests that these samples are
consistent with each other for a given ρ ------ On the relationships between structural properties and packing density
of uniform spheres

Nevertheless, the LS compression protocol is an
excellent tool to approach the jamming point: Besides being
fast, it has been amply verified that it closely reproduces
the (phenomenological) equation of state ------ Hard-sphere jamming through the lens of linear optimization


DEM涉及到软球；
体积模量（B）关注的比较少，可以直接算出来
软球
lammps去掉friction
或LS生成初始结构，压缩使得所有颗粒重叠，再能量最小化

思路：压强与order双向增加PD，
bcc用pathy model，到Corey主页搜K. Zhang，

### Q6
![0cc3bed9e61d25023982eb57a587083](https://user-images.githubusercontent.com/72123149/230899166-92151b09-d185-4297-b127-9d9792b6804f.png)


### RDF
![40b13aad2ffa5c9df6ed0f1b44ee70c](https://user-images.githubusercontent.com/72123149/230897274-be805eee-a792-4dd3-a144-e41fdc92f4a6.png)
Mechanical Characterization of Partially Crystallized Sphere Packings附件里有提到新的RDF计算方法

### Ovito analysis
![2a7e1b28df17fb83e17719d53084c0b](https://user-images.githubusercontent.com/72123149/230897726-7c96fd15-4644-413f-b8fa-dfe61d00948b.png)
![b24c3b8029da86689646918e2b9be96](https://user-images.githubusercontent.com/72123149/230900170-7c298611-d9be-46ac-b2cd-e198868b5066.png)


### voronoi分析
![a7b120b85bdb3dcbb41ea7dbcb013af](https://user-images.githubusercontent.com/72123149/230897848-c0cdd0cb-730e-47df-ace3-6746fa3f2156.png)
![bc8960eb945c11d925d1a326efd5c3e](https://user-images.githubusercontent.com/72123149/230899604-aba76151-5e00-47bd-84d9-cfa994999100.png)


### 配位数
Mechanical Characterization of Partially Crystallized Sphere Packings里的Zg和Zm

![a0f03b613fbd8b4820080777d848b89](https://user-images.githubusercontent.com/72123149/230898436-15c5993a-bcf6-4a6e-811e-82a405a0a85a.png)

![a0a3f9f1ff6c3e5097ab1ae8a61c912](https://user-images.githubusercontent.com/72123149/230898452-d1f5c41f-958e-4256-af21-a89c2ed5caf2.png)

![e178cd3aa6cc81ffa18270e6e79c31d](https://user-images.githubusercontent.com/72123149/230899631-b4053f1e-6555-4db7-ba94-a498c80903c1.png)

![10d1c26f77c907f9185fdb5b6ebbd27](https://user-images.githubusercontent.com/72123149/230900203-3a4bf0e8-8c3c-44ff-afa9-44667cdab28f.png)

![fdf5a6e9f9cf44c92598f523eb6ac87](https://user-images.githubusercontent.com/72123149/230900217-8be238a0-69dc-4814-a522-3c8e474514eb.png)

Zg的计算：
![b5706419ee24e9a3f0c42807afc1f97](https://user-images.githubusercontent.com/72123149/231064797-fb0620bc-3d2d-4760-9ce7-05588800eeb9.png)


### 力链分析
![80a31c54bba1991490be728e0ba7d69](https://user-images.githubusercontent.com/72123149/230898497-cd2f7fa8-c3a7-4a9e-92c7-51c1cabb5c57.png)

### 剪切
![fe81aa33c23a5688c101881c1c0c1fa](https://user-images.githubusercontent.com/72123149/230898638-fca64fd9-836b-4a13-b0e6-03636108c7ea.png)

![15c9707fa4a2f9c4df93d991839599e](https://user-images.githubusercontent.com/72123149/230899669-9acc44b1-6b78-4258-932b-be71f68a7cfa.png)

![62e8233b54157d0062a20e6d220ad74](https://user-images.githubusercontent.com/72123149/230899685-dc16e124-4897-442d-b31b-9e4b3baa8292.png)
这里的本构和dem怎么建模，可以参考Mechanical Characterization of Partially Crystallized Sphere Packings。

### 其他可借鉴
![8e4010017f74ef68be5956d7e7b38c6](https://user-images.githubusercontent.com/72123149/230898995-ba799dc0-dfb7-4eed-81c4-ed6351fcf285.png)
![402ac9d034ec1138b59a4e90b18704a](https://user-images.githubusercontent.com/72123149/230899057-9a6acfc7-9362-4c73-9240-c8ee22af49f2.png)
有篇名为structural and mechanical characteristics of sphere packings near the jamming trainsition值得看一下。


