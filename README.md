# colvR
Package 
Ce package propose des modèles d'imputation pour des données de comptages d'individus prises sur plusieurs sites et plusieurs années. 

Ce package contient principalement 3 fonctions :
  Miss.PLNPCA : Fonction d'imputation dans le cadre d'espèces communes (peu de 0)
  Miss.ZIPLNPCA : Fonction d'imputation dans le cadre d'espèces rares (beaucoup de 0)
  Miss.ZIPLNPCA.logS : Fonction équivalente à la précédente mais parfois plus efficace. 

  Ces trois fonctions ne demandent en entrée que la matrice de comptage Y, la matrice de covariables X et le paramètres q (taille de l'espace latent). 
