#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#define NmbEntree 20
#define NmbCouche 5
#define teta 0.01 // TAUX D'APPRENTISSAGE
#define NombreNeuroneMax 128
#define NmbRetropropagation 3000000
#define SCALE 255
#define Sortie 1
#define SortieParEntrainement 1 // 1 reponse par entrainement (1 lettre)

//=============================================================================================================================================================================================

float entree[NmbEntree];
float SEED;
float Y[SortieParEntrainement]; // OBJECTIF
char ligne[NmbEntree+Sortie+1]; // +Sortie+1 car on veut aussi lire Y et le /0
float ErreurTotalePremierePartie=0;
float ErreurTotaleDeuxiemePartie=0;
float ErreurTotaleTroisiemePartie=0;
float ErreurTotaleQuatriemePartie=0;

typedef struct {
    float Z; // SOMME D'UNE NEURONNE : SOMME DE : (POIDS X ENTREE) + BIAIS
    float A; // SIGMOIDE DE Z : MISE A NIVEAU ENTRE 0 ET 1
    float P[NmbEntree]; // TABLEAU DES POIDS   // LA TAILLE DU TABLEAU EST NmbEntree CAR IL N'Y AURA JAMAIS PLUS D'ENTREE DANS UNE NEURONE QUE LE NOMBRE D'ENTREES DE DEPART
    float B; // BIAIS
    float DELTA; // DELTA : GRADIENT : DC/DZ
} NEURONE;

typedef struct {
    int NmbNeuroneCouche; // NOMBRE DE NEURONES PAR COUCHE (INITIALISE DANS : InitialisationNombreNeuroneCouhe[])
    NEURONE NEURONE_ORDRE[NombreNeuroneMax];
} COUCHE;

COUCHE COUCHE_ORDRE[NmbCouche];

int InitialisationNombreNeuroneCouhe[NmbCouche] = {128, 64, 32, 16, Sortie}; // INITIALISE LE NOMBRE DE NEURONES POUR CHAQUE COUCHE

//=============================================================================================================================================================================================

void Initialisation(){

    // AFFECTE LE NOMBRE DE NEURONES PAR COUCHE
    for(int i=0; i<NmbCouche; i++){
        COUCHE_ORDRE[i].NmbNeuroneCouche = InitialisationNombreNeuroneCouhe[i];
    }

    // INITIALISATION ALEATOIRE DES POIDS ET BIAIS DE LA COUCHE N A LA COUCHE 2
    for(int g=1; g<NmbCouche; g++){
        for(int k=0; k<COUCHE_ORDRE[g].NmbNeuroneCouche; k++){
            for(int i=0; i<COUCHE_ORDRE[g-1].NmbNeuroneCouche; i++){
                COUCHE_ORDRE[g].NEURONE_ORDRE[k].P[i] = (float)rand()/RAND_MAX - 0.5f; // INITALISE UNE VALEUR ENTRE -0.5 et 0.5
            }
            COUCHE_ORDRE[g].NEURONE_ORDRE[k].B = (float)rand()/RAND_MAX - 0.5f; // INITALISE UNE VALEUR ENTRE -0.5 et 0.5
        }
    }

    // INITIALISATION ALEATOIRE DES POIDS ET BIAIS DE LA COUCHE 1
    for(int k=0; k<COUCHE_ORDRE[0].NmbNeuroneCouche; k++){
        for(int i=0; i<NmbEntree; i++){
            COUCHE_ORDRE[0].NEURONE_ORDRE[k].P[i] = (float)rand()/RAND_MAX - 0.5f; // INITALISE UNE VALEUR ENTRE -0.5 et 0.5
        }
        COUCHE_ORDRE[0].NEURONE_ORDRE[k].B = (float)rand()/RAND_MAX - 0.5f; // INITALISE UNE VALEUR ENTRE -0.5 et 0.5
    }

    for(int k=0; k<COUCHE_ORDRE[0].NmbNeuroneCouche; k++){ // INITIALISE POUR LA COUCHE 1
            COUCHE_ORDRE[0].NEURONE_ORDRE[k].A = 0;
            COUCHE_ORDRE[0].NEURONE_ORDRE[k].DELTA = 0;
    }
     for(int g=0; g<NmbCouche; g++){
        for(int k=0; k<COUCHE_ORDRE[g].NmbNeuroneCouche; k++){ // INITIALISE LE RESTE
            COUCHE_ORDRE[g].NEURONE_ORDRE[k].A = 0;
            COUCHE_ORDRE[g].NEURONE_ORDRE[k].DELTA = 0;
        }
    }

}

//=============================================================================================================================================================================================

int InitialisationProchaineLigne(FILE* fichierWikipedia){
   
    int c;
    // On remplit les cases en lisant la suite réelle du fichier
    for (int i = 0; i < NmbEntree; i++) {
        c = fgetc(fichierWikipedia);
        if (c == EOF) {
            rewind(fichierWikipedia);
            c = fgetc(fichierWikipedia);
        }
        ligne[i] = (char)c;
        entree[i] = (float)c / 255.0f;
    }
   
    for(int k=0; k<COUCHE_ORDRE[NmbCouche-1].NmbNeuroneCouche; k++){
        int cible = fgetc(fichierWikipedia);
        if (cible == EOF) {
            rewind(fichierWikipedia);
            cible = fgetc(fichierWikipedia);
        }
        Y[k] = (float)cible / 255.0f;
    }
    return 1;
}

//=============================================================================================================================================================================================


void RetropropagationAvant(){

    // INITIALISATION DE Z ET A POUR LA COUCHE 1
    for(int k=0; k<COUCHE_ORDRE[0].NmbNeuroneCouche; k++){ // ON CALCULE Z/A POUR LA PREMIERE COUCHE  // K CORRESPOND AU NUMERO DE LA NEURONE

        COUCHE_ORDRE[0].NEURONE_ORDRE[k].Z = 0;
        COUCHE_ORDRE[0].NEURONE_ORDRE[k].A = 0;

        for(int i=0; i<NmbEntree; i++){ // I CORRESPOND AU NUMERO DU POIDS ET DE L'ENTREE DE LA NEURONE

            COUCHE_ORDRE[0].NEURONE_ORDRE[k].Z += entree[i] * COUCHE_ORDRE[0].NEURONE_ORDRE[k].P[i]; // ICI LA SOMME Z EST CALCULEE
        }
            COUCHE_ORDRE[0].NEURONE_ORDRE[k].Z += COUCHE_ORDRE[0].NEURONE_ORDRE[k].B; // ON RAJOUTE LA CONSTANTE BIAIS A LA FIN
            COUCHE_ORDRE[0].NEURONE_ORDRE[k].A = 1.0f/(1.0f + expf(-(COUCHE_ORDRE[0].NEURONE_ORDRE[k].Z))); // UNE FOIS LE Z CALCULE ON PEUT CALCULER A
    }

    // INITIALISATION DES COUCHES 2 A LA COUCHE N-1
    for(int g=1; g<NmbCouche; g++){ // ON REPETE LE PROCESSUS DE LA COUCHE 2 A LA COUCHE N (C'est pour ca qu'on part de g=1)

        for(int k=0; k<COUCHE_ORDRE[g].NmbNeuroneCouche; k++){ // NOMBRE DE NEURONES PAR COUCHE

            COUCHE_ORDRE[g].NEURONE_ORDRE[k].Z = 0;

            for(int i=0; i<COUCHE_ORDRE[g-1].NmbNeuroneCouche; i++){ // I CORRESPOND AU NUMERO DU POIDS ET DE L'ENTREE DE LA NEURONE // coucheTableau[0].NmbNeuroneCouche correspond au nombre d'entrée de la couche car : nombre de neurone L = nombre entree couche L+1

                COUCHE_ORDRE[g].NEURONE_ORDRE[k].Z += COUCHE_ORDRE[g-1].NEURONE_ORDRE[i].A * COUCHE_ORDRE[g].NEURONE_ORDRE[k].P[i];
            }
            COUCHE_ORDRE[g].NEURONE_ORDRE[k].Z += COUCHE_ORDRE[g].NEURONE_ORDRE[k].B;

            COUCHE_ORDRE[g].NEURONE_ORDRE[k].A = 1.0f/(1.0f + expf(-COUCHE_ORDRE[g].NEURONE_ORDRE[k].Z));
        }
    }

}

//=============================================================================================================================================================================================

void RetropropagationArriere(){
    // ON POSE C = 0.5(A^N - Y)²

    // CALCUL DU GRADIENT DE LA COUCHE N
    // COMME A^N EST UNE VARIABLE DE C, LA DERIVEE EST SIMPLE
    for(int k=0; k<COUCHE_ORDRE[NmbCouche-1].NmbNeuroneCouche; k++){ // ON UTILISE NmbCouche-1 CAR LE TABLEAU DECALE TOUS DE 1 CAR LA PREMIER TERME (COUCHE 1), EST LE TERME 0. DONC LA COUCHE N EST LE TERME N-1

        COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].DELTA = 0;

        COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].DELTA = (COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].A - Y[k]); // ON ONTULISE PAS DE SIGMOIDE POUR LA COUCHE N CAR ON A PAS BESOIN DE REDUIRE LE RESULTAT ENTRE 0 ET 1

        for(int i=0; i<COUCHE_ORDRE[NmbCouche-2].NmbNeuroneCouche; i++){ // ACTUALISER TOUT LES NOUVEAUX POIDS ET BIAIS

            COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].P[i] -= teta * COUCHE_ORDRE[NmbCouche-2].NEURONE_ORDRE[i].A * COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].DELTA;
        }
        COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].B -= teta * COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].DELTA;
    }

    // CALCUL DU GRADIENT DE LA COUCHE N-1 A LA COUCHE 2
    // COMME A != A^N, LA DERIVEE EST DEPENDANTE DU PRECEDENT GRADIENT (DELTA)
    for(int g=NmbCouche-2; g>0; g--){ // ON DEMARRE A N-1 PUIS ON DESCENDS JUSQU'A LA COUCHE 2

        for(int k=0; k<COUCHE_ORDRE[g].NmbNeuroneCouche; k++){ // LE NOMBRE DE NEURONE PAR COUCHE A PARCOURIR

            COUCHE_ORDRE[g].NEURONE_ORDRE[k].DELTA = 0;

            for(int j=0; j<COUCHE_ORDRE[g+1].NmbNeuroneCouche; j++){ // FAIRE LA SOMME DE TOUTES LES NEURONES DE LA COUCHE SUPERIEUR MULTIPLIé PAR LEUR MATRICE DE POIDS

                COUCHE_ORDRE[g].NEURONE_ORDRE[k].DELTA += COUCHE_ORDRE[g+1].NEURONE_ORDRE[j].DELTA * COUCHE_ORDRE[g+1].NEURONE_ORDRE[j].P[k];
            }

            COUCHE_ORDRE[g].NEURONE_ORDRE[k].DELTA = COUCHE_ORDRE[g].NEURONE_ORDRE[k].DELTA * (COUCHE_ORDRE[g].NEURONE_ORDRE[k].A * (1.0f - COUCHE_ORDRE[g].NEURONE_ORDRE[k].A));

            for(int i=0; i<COUCHE_ORDRE[g-1].NmbNeuroneCouche; i++){ // ACTUALISER TOUT LES NOUVEAUX POIDS ET BIAIS

                COUCHE_ORDRE[g].NEURONE_ORDRE[k].P[i] -= teta * COUCHE_ORDRE[g-1].NEURONE_ORDRE[i].A * COUCHE_ORDRE[g].NEURONE_ORDRE[k].DELTA;
            }
            COUCHE_ORDRE[g].NEURONE_ORDRE[k].B -= teta * COUCHE_ORDRE[g].NEURONE_ORDRE[k].DELTA;
        }
    }

    // CALCUL DU GRADIENT DE LA COUCHE 1  (calulé séparement car son entrée est entree[NmbEntree])
    for(int k=0; k<COUCHE_ORDRE[0].NmbNeuroneCouche; k++){ // LE NOMBRE DE NEURONE DE LA COUCHE A PARCOURIR

        COUCHE_ORDRE[0].NEURONE_ORDRE[k].DELTA = 0;

        for(int j=0; j<COUCHE_ORDRE[1].NmbNeuroneCouche; j++){ // FAIRE LA SOMME DE TOUTES LES NEURONES DE LA COUCHE 2 MULTIPLIé PAR LEUR MATRICE DE POIDS

            COUCHE_ORDRE[0].NEURONE_ORDRE[k].DELTA += COUCHE_ORDRE[1].NEURONE_ORDRE[j].DELTA * COUCHE_ORDRE[1].NEURONE_ORDRE[j].P[k];
        }
        COUCHE_ORDRE[0].NEURONE_ORDRE[k].DELTA = COUCHE_ORDRE[0].NEURONE_ORDRE[k].DELTA * (COUCHE_ORDRE[0].NEURONE_ORDRE[k].A * (1.0f - COUCHE_ORDRE[0].NEURONE_ORDRE[k].A));

        for(int i=0; i<NmbEntree; i++){ // ACTUALISER TOUT LES NOUVEAUX POIDS ET BIAIS

            COUCHE_ORDRE[0].NEURONE_ORDRE[k].P[i] -= teta * entree[i] * COUCHE_ORDRE[0].NEURONE_ORDRE[k].DELTA;
        }
        COUCHE_ORDRE[0].NEURONE_ORDRE[k].B -= teta * COUCHE_ORDRE[0].NEURONE_ORDRE[k].DELTA;
    }

}

//=============================================================================================================================================================================================

// void AfficherResultat(){
//     for(int g=0; g<NmbCouche; g++){
//         for(int k=0; k<COUCHE_ORDRE[NmbCouche-1].NmbNeuroneCouche; k++){
//             printf("\n");
//             printf("%f", COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].DELTA);
//             // printf("\n");
//         }
//     }
// }

int main(){
    srand(time(NULL));
    Initialisation();

    FILE* fichierWikipedia = fopen("CORPUS.txt", "r"); // OUVERTURE LECTURE
    if(fichierWikipedia == NULL){
        perror("\nErreur lors de l'ouverture du fichier : ");
    }
    printf("\nFichier ouvert avec succes.\n");

    for(int r=0; r<NmbRetropropagation; r++){
        InitialisationProchaineLigne(fichierWikipedia);
        RetropropagationAvant();
        RetropropagationArriere();
        if(r % 10000 == 0){
            printf("\n\n");
            for(int k=0; k<COUCHE_ORDRE[NmbCouche-1].NmbNeuroneCouche; k++){
                printf("Cycle: %d, ||| Erreur de la sortie %d: %f\n", r, k, COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].DELTA);

                if(r < (NmbRetropropagation * 0.25)){
                ErreurTotalePremierePartie += COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].DELTA;
                }
                else if (r < (NmbRetropropagation * 0.5)){
                    ErreurTotaleDeuxiemePartie += COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].DELTA;
                }
                else if (r < (NmbRetropropagation * 0.75)){
                    ErreurTotaleTroisiemePartie += COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].DELTA;
                }
                else{
                    ErreurTotaleQuatriemePartie += COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].DELTA;
                }
            }
        }
    }
    printf("\n\n===================================");
    printf("\n\nVOICI LA SUITE LOGIQUE : \n\n");
    // AFFICHER LES ENTREES
    for(int i=0; i<NmbEntree; i++){
        printf("%c", (unsigned char)(entree[i]*255)); // ON REMET EN LETTRES
    }
    for(int k=0; k<COUCHE_ORDRE[NmbCouche-1].NmbNeuroneCouche; k++){
        printf("\n\nVOICI LA CIBLE %d (LA LETTRE MANQUANTE DE LA SUITE LOGIQUE) : %c\n\nVOICI CE QUE L'INTELLIGENCE ARTIFICIELLE A PREDIT : %c\nVOICI L'ERREUR : %f\n\n",
                k, (unsigned char)(Y[k]*255), (unsigned char)(COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].A*255), COUCHE_ORDRE[NmbCouche-1].NEURONE_ORDRE[k].DELTA);
        printf("\n\nErreur totale du premier quart : %f\n\nErreur totale du deuxieme quart : %f\n\nErreur totale du troisieme quart : %f\n\nErreur totale du quatrieme quart : %f\n\n",
                ErreurTotalePremierePartie, ErreurTotaleDeuxiemePartie, ErreurTotaleTroisiemePartie, ErreurTotaleQuatriemePartie);
        printf("\n\n===================================\n\n");
    }

    fclose(fichierWikipedia);
}
