# Projet stochastique : Vol d'étrouneaux

## Description
Modélisation des effets de groupe dans les vols d'étourneaux avec le modèle de Viscek.

Sujet du projet disponible ainsi que tout les codes et Data sont idsponible [ici](https://github.com/C2gouttesdeau/Projet_stochastique)

## Fichiers

Le projet contient les fichiers suivants :

- `exploitation.py` : C'est le point d'entrée du programme. Ce script permet d’exécuter le script main.py pour plusieurs cas de figure. C’est ici que les variables sont définie.

- `main.py` :  Il exécute la simulation et appelle les fonctions d'analyse et de visualisation. Ce script traite le problème, il permet de faire plusieurs itérations pour une liste sur N le nombre de particules et sur η la variable aléatoire.

- `routines.py` : il s'agit du fichier où toutes les fonctions sont définies.



## Utilisation et paramètre du simulateur
Pour exécuter le programme exécuter le fichier `exploitation.py` dans un environnement pyhton. Tout les paramètres se trouve dans ce fichier. 

- Les variables du problème sont explicité dans le code et dans le sujet.

- `Animation` ce booléen permet de créer l'annimation des postions des particules au court du temps.
- `Trajectoires` ce boléen permet d'afficher la trajectoire d'une des particules au court du temps dans l'annimation
- `Analyse` ce booléen permet de réaliser l'analyse de la simulation avec la mise en place de plusieurs affichage d'écrit ci-après.
- `Save` ce booléen permet de sauvegarder toute les données, graphique et animation de la simulation. Ces données sont enregistré dans un dossier portant le nom de la simualtion (valeurs des paramètre). 

- `Auto_stop` ce booléen permet d'arrêter automatique la silation l'orsque que la fonction `Phi`(d'écrite dans le sujet) est constante en moyenne.

-  `coef_dir_limit` est la valeur de la pente de la régression linéaire sur la moyenne de `Phi`.

- `Perio` est un booléen qui permet ou non d'appliquer les conditions aux limites périodiques. 

- `Size_max` permet de déterminer la taille maximale des particules dans l'annimation (Pour N=1). `N_max` donne la valeur de N pour la taille la plus petite (`Size`=10). La taille est proportionnelle pour un nombre de particules compris entre 0 et `N_max` et est de 10 pour des taille supérieur.