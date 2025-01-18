import pandas as pd

def precision_recall(tab):
    relevant_recommendations = tab["reaction"].apply(lambda x: 1 if x == "like" else 0)
    recommendations = tab["reaction"].apply(lambda x: 1 if x else 0)
    accuracy = relevant_recommendations.sum() / recommendations.sum() if not recommendations.empty else 0
    return accuracy

if __name__=="__main__":
    tab = pd.read_csv("tab_AB.csv")
    tab_knn = tab[tab["model"] == "knn"]
    tab_neuMF = tab[tab["model"] == "ncf"]
    knn_accuracy = precision_recall(tab_knn)
    print("Dokładność modelu knn: ", knn_accuracy)
    neuMF_accuracy = precision_recall(tab_neuMF)
    print("Dokładność modelu ncf: ", neuMF_accuracy)


