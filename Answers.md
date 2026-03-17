### Answers Examen OS202 2026 - Luiza Gonçalves Soares  

### Answer – Question préliminaire
---

Les galaxies ont généralement une forme aplatie, comme un disque. La majorité des étoiles se trouve dans le plan \(Oxy\), et il y a très peu de variation selon l’axe \(Oz\).  

Donc, augmenter \(N_k\) (le nombre de cellules selon \(Oz\)) n’est pas vraiment utile, car la plupart des cellules seraient vides ou contiendraient très peu d’étoiles.  

Cela augmente le coût de calcul sans améliorer vraiment la précision.  

Ainsi, prendre \(N_k = 1\) est suffisant et plus efficace.  


### Answer – Mesure du temps initial  
---
La partie la plus intéressante à paralléliser est le calcul des trajectoires.  

En effet :  

1. Le calcul des accélérations et la mise à jour des positions demandent beaucoup d’opérations et prennent la majorité du temps d’exécution. Sans optimisation, la complexité est proche de \(O(N^2)\).  

2. L’affichage dépend surtout du GPU et du rafraîchissement de l’écran, donc le paralléliser sur CPU n’apporte pas beaucoup de gain.  

3. Selon la loi d’Amdahl, améliorer la partie qui prend le plus de temps (le calcul) permet d’avoir le plus grand gain global.  


### Answer - Parallélisation en numba du code séquentiel
---

Pour paralléliser le code avec Numba, l’option `parallel=True` est ajoutée aux fonctions décorées avec `@njit`, et `range` est remplacé par `prange` dans les boucles indépendantes, notamment la boucle principale sur les étoiles pour le calcul des accélérations.

Afin d’évaluer les performances, le programme est exécuté avec différents nombres de threads en utilisant la variable d’environnement `NUMBA_NUM_THREADS`. Pour garantir une comparaison cohérente, les temps mesurés au même instant (step 60) sont utilisés.

| Nombre de threads | Temps moyen par pas (ms) | Accélération |
|---|---:|---:|
| 1 | 231.26 | 1.00 |
| 2 | 158.48 | 1.46 |
| 3 | 175.08 | 1.32 |
| 4 | 205.37 | 1.13 |

L’accélération est calculée avec la formule suivante :

\[
S(p) = \frac{T_1}{T_p}
\]

où \(T_1\) est le temps avec 1 thread et \(T_p\) le temps avec \(p\) threads.

On observe que la meilleure performance est obtenue avec 2 threads, avec une accélération d’environ 1.46. Avec 3 threads, le gain reste positif mais plus faible. Avec 4 threads, le gain est limité.

Cela montre que la parallélisation avec Numba améliore les performances, mais que le gain n’est pas linéaire, notamment à cause des surcoûts liés à la gestion des threads et au fait que certaines parties du code restent séquentielles.

### Answer - Separation de l`affichage et du calcul
---
Dans cette version, MPI est utilisé pour séparer l’affichage et le calcul sur deux processus :

- le rang 0 est dédié à l’affichage ;
- le rang 1 effectue le calcul des positions des étoiles.

Le rang 1 envoie les positions mises à jour au rang 0, qui les utilise pour mettre à jour l’affichage. Cette approche permet de découpler les deux tâches et d’éviter que l’affichage bloque directement le calcul.

Les temps observés sont les suivants :

| Threads Numba | Temps de mise à jour observé |
|---|---:|
| 2 | 194 ms |
| 4 | 191 ms |
| 8 | 209 ms |

On observe que les performances restent proches avec 2 et 4 threads, avec un léger avantage pour 4 threads. En revanche, avec 8 threads, le temps augmente.

Cela montre que la séparation entre affichage et calcul fonctionne, mais que le gain reste limité par le coût des communications MPI et par le surcoût lié à un trop grand nombre de threads. Dans cette configuration, 4 threads donne le meilleur résultat parmi les tests observés.



### Answers - Parallélisation du calcul
#### Answer A
---

Les performances ont été mesurées en faisant varier le nombre de processus MPI et le nombre de threads Numba. La configuration de référence est (1 processus, 1 thread).

| Processus MPI | Threads Numba | Temps total (s) | Accélération |
|---|---:|---:|---:|
| 1 | 1 | 17.175 | 1.000 |
| 1 | 2 | 17.720 | 0.969 |
| 1 | 4 | 18.925 | 0.907 |
| 2 | 1 | 11.700 | 1.468 |
| 2 | 2 | 12.218 | 1.406 |
| 2 | 4 | 13.401 | 1.282 |
| 4 | 1 | 10.599 | 1.620 |
| 4 | 2 | 12.151 | 1.413 |
| 4 | 4 | 18.865 | 0.910 |

Les résultats montrent que :

- La meilleure performance est obtenue avec 4 processus MPI et 1 thread Numba, avec une accélération d’environ 1.62.
- L’augmentation du nombre de threads Numba n’améliore pas les performances dans ce cas. Au contraire, avec 4 threads, les performances se dégradent fortement.
- Cela s’explique par le surcoût de gestion des threads, la synchronisation, ainsi que le fait que plusieurs processus MPI utilisent déjà les ressources du processeur.
- Lorsque le nombre total de threads dépasse le nombre de cœurs disponibles, on observe une dégradation des performances (oversubscription).

Ainsi, dans cette configuration, il est plus efficace d’augmenter le nombre de processus MPI que le nombre de threads Numba.


### Answer B
---

La densité d’étoiles est plus élevée près du trou noir et diminue avec la distance. Avec une distribution uniforme des cellules entre les processus, certains processus reçoivent des zones très denses et doivent traiter beaucoup plus d’étoiles, tandis que d’autres ont beaucoup moins de travail. 

Cela crée un déséquilibre de charge : certains processus terminent rapidement et doivent attendre les autres lors des synchronisations MPI. Ce déséquilibre limite l’accélération globale du programme. De plus, les communications et synchronisations entre processus ajoutent un surcoût qui réduit encore les performances.


### Answer C
---

Une meilleure stratégie consiste à répartir les cellules non pas de manière uniforme, mais en fonction de la charge de calcul.  

Par exemple, les processus proches du centre de la galaxie, où la densité d’étoiles est élevée, pourraient recevoir moins de cellules, tandis que les processus situés en périphérie pourraient en recevoir davantage.  

De manière générale, l’objectif est d’équilibrer le nombre d’étoiles ou le coût de calcul entre les processus afin de réduire le déséquilibre de charge.

Cependant, cette distribution intelligente introduit un nouveau problème de performance. Les domaines deviennent irréguliers, ce qui peut augmenter le nombre de communications entre processus.  

En particulier :
- le nombre de frontières entre processus augmente,
- le volume de données des cellules fantômes à échanger devient plus important,
- l’implémentation devient plus complexe.

Ainsi, il existe un compromis entre un meilleur équilibrage du calcul et un coût de communication plus élevé, qui peut limiter le gain global de performance.



## Answers - Pour aller plus loin

### Distribution des boîtes et sous-boîtes entre processus
---

Une solution simple consiste à utiliser la structure du quadtree pour répartir le travail.  

Les niveaux supérieurs de l’arbre peuvent être partagés entre tous les processus, car ils contiennent peu de données et sont utilisés par tout le monde.  

Ensuite, à partir d’un certain niveau, les sous-boîtes sont distribuées entre les processus. Chaque processus s’occupe d’une zone de l’espace (un groupe de sous-boîtes proches).  

Cela permet de garder la majorité du calcul local et de limiter les communications.


### Parallélisation MPI du calcul de l’accélération
---

À chaque pas de temps :

1. chaque processus met à jour les étoiles dont il est responsable ;
2. les niveaux supérieurs de l’arbre sont partagés entre les processus ;
3. chaque processus calcule l’accélération pour ses étoiles :
   - si une boîte est loin, on utilise sa masse et son centre de masse ;
   - sinon, on descend dans les sous-boîtes ;
4. si des données d’un autre processus sont nécessaires, elles sont échangées avec MPI ;
5. les positions et vitesses sont mises à jour localement ;
6. les étoiles qui changent de zone sont envoyées au bon processus.

Cette méthode permet de réduire le coût du calcul et de paralléliser efficacement le travail, même si des communications entre processus restent nécessaires.