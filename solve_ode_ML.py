import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

# Définition de l'ODE
def ode_function(x):
    # return -2 * x * np.exp(-x**2) + 2 * x**2 * np.exp(-x**2)
    # return np.log(1/x) * 3
    return np.cos(x)


# def ode_function2(x):
#     return -2 * np.exp(-x**2) + 2 * x**2 * np.exp(-x**2) + np.sin(x)


# Génération de données d'entraînement avec l'enseignant
def generate_data(num_samples):
    x_train = np.linspace(0.00001, 2, num_samples)
    y_train = ode_function(x_train)  # Solution analytique de l'ODE
    return x_train, y_train


# Définition de l'architecture du réseau de neurones étudiant
def create_student_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

# Entraînement du modèle étudiant
def train_student(student_model, x_train, y_train, num_epochs):
    student_model.compile(optimizer='adam', loss='mean_squared_error')
    student_model.fit(x_train, y_train, epochs=num_epochs, verbose=0)
    return student_model


# Évaluation du modèle étudiant sur des données de test
def evaluate_student(student_model, x_test):
    return student_model.predict(x_test)

# Visualisation des résultats
def plot_results(x_train, y_train, x_test, y_pred):
    plt.plot(x_train, y_train, label='Données d\'entraînement')
    plt.scatter(x_test, y_pred, color='red', label='Prédiction de l\'étudiant')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Résolution d\'une ODE avec un réseau de neurones étudiant')
    plt.legend()

# Paramètres
num_samples = 100
num_epochs = 50

# Génération de données et initialisation du modèle étudiant
x_train, y_train = generate_data(num_samples)
x_train = x_train.reshape(-1, 1)  # Reshape pour l'entrée du modèle
student_model = create_student_model()

# Entraînement du modèle étudiant
train_student(student_model, x_train, y_train, num_epochs)

# Évaluation du modèle étudiant sur l'ensemble de test
x_test = np.linspace(0.000001, 2, 100).reshape(-1, 1)
x_test = np.random.uniform(0.000001, 2, 100).reshape(-1,1)
y_pred_1 = evaluate_student(student_model, x_test)



def teacher_2(x_train):
    # return -2 * np.exp(-x_train**2) + 2 * x_train**2 * np.exp(-x_train**2) + np.sin(x_train) + x_train**2
    return np.sin(x_train)

# Génération de données d'entraînement pour le deuxième enseignant
def generate_data_second_teacher(num_samples):
    x_train = np.linspace(0.000001, 2, num_samples)
    y_train = -2 * np.exp(-x_train**2) + 2 * x_train**2 * np.exp(-x_train**2) + np.sin(x_train) + x_train**2  # Solution analytique de l'ODE avec une perturbation sinusoïdale
    return x_train, y_train

# Entraînement du modèle étudiant en ne modifiant que les poids de la tête
def train_student_head(student_model, x_train, y_train, num_epochs):
    student_model.layers[-1].trainable = True  # Ne modifier que les poids de la tête
    student_model.compile(optimizer='adam', loss='mean_squared_error')
    student_model.fit(x_train, y_train, epochs=num_epochs, verbose=0)
    return student_model

# Paramètres
num_samples = 100
num_epochs = 50

# Génération de données pour le deuxième enseignant
x_train_second_teacher, y_train_second_teacher = generate_data_second_teacher(num_samples)
x_train_second_teacher = x_train_second_teacher.reshape(-1, 1)  # Reshape pour l'entrée du modèle

# # Entraînement du modèle étudiant en ne modifiant que les poids de la tête
train_student_head(student_model, x_train_second_teacher, y_train_second_teacher, num_epochs)

# # Évaluation du modèle étudiant sur l'ensemble de test
# y_pred_second_teacher = second_teacher_model.predict(x_test)

y_pred_student_head = student_model.predict(x_test)


# Visualisation des résultats
# plot_results(x_train, y_train, x_train, evaluate_student(student_model, x_train))
# plot_results(x_train, y_train, x_test, y_pred_1)
# plt.show()


# # Visualisation des résultats
# plt.figure(figsize=(10, 6))
# plt.plot(x_train, y_train, label='Données d\'entraînement premier enseignant - Tâche A')
# plt.scatter(x_test, y_pred_1, color='dodgerblue', label='Prédiction de l\'étudiant tâche A (corps et tête)')
# plt.plot(x_train, y_train_second_teacher, color='green', label='Données d\'entrainement deuxième enseignant - Tâche B')
# plt.scatter(x_test, y_pred_student_head, color='lightblue', label='Prédiction de l\'étudiant tâche B (tête seulement)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Comparaison des prédictions enseignants et de l\'étudiant (entraînement tête seulement)')
# plt.legend()
# plt.show()

y_train_second_teacher = teacher_2(x_train)

import pickle

def modele_test(nb_epoch, state_change, x_train, y_train1, y_train2):
    modele = create_student_model()
    filename = 'model_base.sav'
    pickle.dump(modele, open(filename, 'wb'))
    for i in range(1,nb_epoch):
        if i <= state_change:
            modele = train_student(modele, x_train, y_train1, i)
            filename = 'model_eq1_' + str(i) + '.sav'
            pickle.dump(modele, open(filename, 'wb'))
        else : 
            j = i - state_change
            modele = train_student_head(modele, x_train, y_train2, j)
            filename = 'model_eq2_' + str(j) + '.sav'
            pickle.dump(modele, open(filename, 'wb'))


modele_test(50, 25, x_train,y_train, y_train_second_teacher)
            

# pickled_model = pickle.load(open('model_base.sav', 'rb'))
# y_predit_base = pickled_model.predict(x_test)
# pickled_model = pickle.load(open('model_eq1_15.sav', 'rb'))
# y_predit_15 = pickled_model.predict(x_test)
# pickled_model = pickle.load(open('model_eq1_25.sav', 'rb'))
# y_predit_25 = pickled_model.predict(x_test)

# pickled_model = pickle.load(open('model_base.sav', 'rb'))
# y_predit_base = pickled_model.predict(x_test)
# pickled_model = pickle.load(open('model_eq1_15.sav', 'rb'))
# y_predit_15 = pickled_model.predict(x_test)
# pickled_model = pickle.load(open('model_eq1_25.sav', 'rb'))
# y_predit_25 = pickled_model.predict(x_test)

# pickled_model = pickle.load(open('model_eq2_1.sav', 'rb'))
# y_predit_2_1 = pickled_model.predict(x_test)
# pickled_model = pickle.load(open('model_eq2_15.sav', 'rb'))
# y_predit_2_15 = pickled_model.predict(x_test)
# pickled_model = pickle.load(open('model_eq2_24.sav', 'rb'))
# y_predit_2_24 = pickled_model.predict(x_test)

# # plt.scatter(x_test, y_predit_base, label = 'base')
# # plt.scatter(x_test, y_predit_15, label = "Tâche A - 15")
# # plt.scatter(x_test, y_predit_25, label = "Tâche A - 25")

# # # plt.scatter(x_test, y_predit_2_1, label = "Tâche B - 1")
# # # plt.scatter(x_test, y_predit_2_15, label = "Tâche B - 15")
# plt.scatter(x_test, y_predit_2_24, label = "Tâche B - 24")

# # # plt.plot(x_train, y_train, label = "Tâche A")
# plt.plot(x_train, y_train_second_teacher, label = "Tâche B")

# plt.legend()
# plt.show()


from sklearn.metrics import mean_squared_error

y_true_1 = ode_function(x_test)
y_true_2 = teacher_2(x_test)

# # x_de_test = np.random.uniform(0, 2, 50)
# # valeurs_y = teacher_2(x_de_test)
# # pickled_model = pickle.load(open('model_eq2_24.sav', 'rb'))
# # valeurs_pred = pickled_model.predict(x_de_test)

# # print(valeurs_y)
# # print(valeurs_pred)

# print(mean_squared_error(y_true_2, y_predit_2_24))

# # # # print('y_true', y_true_1)
# # # print(len(y_true_1))
# # # # print('y_base', y_predit_base)
# # # print(len(y_predit_base))

# # print(mean_squared_error(y_true_2, y_predit_2_24))
# # print(mean_squared_error(y_true_2, y_predit_25))

erreurs_1 = []
erreurs_2 = []

for i in range(1,50):
    if i <= 25 :
        nom1 = 'model_eq1_' + str(i) + '.sav'
        pickled_model1 = pickle.load(open(nom1, 'rb'))
        y_predit1 = pickled_model1.predict(x_test)

        erreurs_1.append(mean_squared_error(y_true_1, y_predit1))
        erreurs_2.append(mean_squared_error(y_true_2, y_predit1))
        
    else :
        nom2 = 'model_eq2_' + str(i-25) + '.sav'
        pickled_model2 = pickle.load(open(nom2, 'rb'))
        y_predit2 = pickled_model2.predict(x_test)

        erreurs_1.append(mean_squared_error(y_true_1, y_predit2))
        erreurs_2.append(mean_squared_error(y_true_2, y_predit2))
    

print(erreurs_1)
# print(len(erreurs_2))
# print(len(np.linspace(1,49,49)))
# print(np.linspace(1,49,49))

plt.scatter(np.linspace(1, 49, 49), erreurs_1, label= 'A')
plt.scatter(np.linspace(1,49,49), erreurs_2, label = 'B')
plt.legend()
plt.show()