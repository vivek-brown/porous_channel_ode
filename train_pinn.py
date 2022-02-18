import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

from pinn import PINN_Model
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
# It's also possible to use the reduced notation by directly setting font.family:
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})
import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)


# @tf.function(jit_compile=True)
# def train(don_model, X_func, X_loc, y):
#     with tf.GradientTape() as tape:
#         y_hat  = don_model(X_func, X_loc)
#         loss   = don_model.loss(y_hat, y)[0]
#
#     gradients = tape.gradient(loss, don_model.trainable_variables)
#     don_model.optimizer.apply_gradients(zip(gradients, don_model.trainable_variables))
#     return(loss)

def main():
    Par = {}




    Par['address'] = 'pinn_models'

    print(Par['address'])
    print('------\n')

    #Reynold's number is fixed here
    Par['Re'] = 10**-3


    x_f = np.random.uniform(-1,1,201)[:,None]
    x_test = np.random.uniform(-1,1,401)[:,None]
    # Par['x_f'] = x_f



    pinn_model = PINN_Model(Par)
    n_epochs = 10000
    batch_size = 50

    tensor = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)

    print("PINN Training Begins")
    begin_time = time.time()


    for i in range(n_epochs+1):
        for end in np.arange(batch_size, x_f.shape[0]+1, batch_size):
            start = end - batch_size
            losses = pinn_model.train_step(tensor(x_f[start:end]))

        if i%100 == 0:

            pinn_model.save_weights(Par['address'] + "/model_"+str(i))

            pde_loss = losses[0].numpy()
            data_loss = losses[1].numpy()
            train_loss = losses[2].numpy()

            u_tilde = pinn_model(tensor(x_test))

            print("epoch:" + str(i) + ", PDE Loss:" + "{:.3e}".format(pde_loss)+ ", Data Loss:" + "{:.3e}".format(data_loss)+ ", Train Loss:" + "{:.3e}".format(train_loss) +  ", elapsed time: " +  str(int(time.time()-begin_time)) + "s"  )

            pinn_model.index_list.append(i)
            pinn_model.train_loss_list.append(train_loss)
            # pinn_model.val_loss_list.append(val_loss)

    #Convergence plot
    index_list = pinn_model.index_list
    train_loss_list = pinn_model.train_loss_list
    # val_loss_list = pinn_model.val_loss_list
    np.savez(Par['address']+'/convergence_data', index_list=index_list, train_loss_list=train_loss_list)


    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    # plt.plot(index_list, val_loss_list, label="val", linewidth=2)
    plt.legend(fontsize=16)
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.savefig(Par["address"] + "/convergence.png")
    plt.close()

    x_test = np.linspace(0,1,401)[:,None]
    x_test = tensor(x_test)
    #Loading best pinn model
    pinn_model_number = index_list[int(np.argmin(np.array(train_loss_list)))]
    print('Best PINN model: ', pinn_model_number)
    pinn_model_address = "pinn_models/model_"+str(pinn_model_number)
    pinn_model.load_weights(pinn_model_address)

    u_pred = pinn_model(x_test)
    temp = pinn_model.PDE_loss(x_test)
    u_x = temp[0]
    u_xx = temp[1]
    u_xxx = temp[2]
    u_xxxx = temp[3]

    plt.close()
    fig = plt.figure(figsize=(7,11))
    plt.plot(x_test, u_pred, linewidth=2, label='f')
    plt.plot(x_test, u_x, linewidth=2 , label="f'")
    # plt.plot(x_test, u_xx, linewidth=2 , label="f''")
    # plt.plot(x_test, u_xxx, linewidth=2 , label="f'''")
    # plt.plot(x_test, u_xxxx, linewidth=2 , label="f''''")
    plt.legend(fontsize=16)
    plt.xlabel('y*', fontsize=18)
    plt.ylabel("f,f' ", fontsize=18)
    plt.title('Re = 0.001', fontsize=20)
    plt.grid()
    plt.savefig('true_vs_prediction.png', dpi=500)
    plt.close()


    print('Complete')


main()
