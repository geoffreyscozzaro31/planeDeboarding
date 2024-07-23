"""
Règles concernant le débarquement des passagers :

**Hypothèses**

* Coordonnées x/y, où x représente le déplacement sur la rangée et y sur le couloir
* Un seul couloir et deux rangées de 3 sièges chacune
* Sortie uniquement par l'avant de l'appareil
* Chaque action (déplacement horizontal, vertical, récupération de bagage) a une durée spécifique à son type

**Algorithme**

À chaque étape :

1. Itérer sur les passagers, par ordre de priorité suivant (ex commençant par les deux premières rangées de l'avion) :

5  3  1    2  4  6

11 9  7    8 10 12
...

3. Pour chaque passager :


    a. Si le passager est assis sur le siège le plus proche du couloir et que le couloir est libre sur cette rangée :
        i. Déplacer le passager d'une case vers la gauche ou la droite (selon la rangée) en utilisant la fonction `x +/- 1`.

    b. Si le passager est assis sur un autre siège et que le siège adjacent côté couloir est libre :
        i. Déplacer le passager d'une case vers la gauche ou la droite en utilisant la fonction `x +/- 1`.

    c. Si le passager vient de se lever et qu'il a un bagage à main :
        i. Le passager récupère son bagage, ce qui le rend immobile pendant une durée `x` secondes (on suppose qu'il range son bagage au niveau de sa rangée).

    d. Si le passager n'a pas de bagage ou si son bagage a été récupéré et que la case `y-1` est libre :
        i. Déplacer le passager vers la case `y-1` à une vitesse définie.


(Vérifier si les règles définies par le papier Schultz sont couvertes par celles enoncées ci-dessus)


**Optimisation potentielle**

Implémenter logique `next-action` pour chaque passager qui calcule le delta minimum pendant lequel aucune action ne
sera effectuée par le passager `i`. Cela permet d'éviter de traiter le passager à chaque pas de temps,
 ce qui optimise la simulation à événements discrets.
"""
