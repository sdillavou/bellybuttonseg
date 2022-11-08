from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Input, Model

def model_S_half_minimum(model_num):
    
    if model_num == 7:
        return 9
    else:
        raise Exception('Model Number Not Recognized.')
        

# Create a keras sequential NN that will train using the input train_gen (training data generator). 
def generate_network(input_shape,model_num=7):
    
    clear_session()

    #create model
    model = Sequential()
    
    
    #add model layers
    if not input_shape[0]>= 2*model_S_half_minimum(model_num):
        raise Exception('Model '+str(model_num)+' requires S_half >= ',str(model_S_half_minimum(model_num)),'.')
    
    if model_num == 7:

        inputs = Input(shape=input_shape, name="img")
        x = layers.Conv2D(64, 3, activation="relu")(inputs)
        x = layers.Conv2D(32, 3, activation="relu")(x)           
        block_1_output = layers.MaxPooling2D(pool_size=(2,2))(x)

        x = layers.Conv2D(32, 3, activation="relu", padding="same")(block_1_output)
        x = layers.Dropout(.1)(x)
        x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        block_2_output = layers.add([x, block_1_output])
        block_2_output = layers.MaxPooling2D(pool_size=(2,2))(block_2_output)
                   
        x = layers.Conv2D(32, 3, activation="relu", padding="same")(block_2_output)
        x = layers.Dropout(.1)(x)
        x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        block_3_output = layers.add([x, block_2_output])
        block_3_output = layers.MaxPooling2D(pool_size=(2,2))(block_3_output)

        x = layers.Flatten()(block_3_output)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(.1)(x)
        outputs2 = layers.Dense(1,activation="relu", name="distance_output")(x)
        
        x2 = layers.add([x,outputs2])
        outputs = layers.Dense(2,activation="softmax",name="category_output")(x2)


        model = Model(inputs, [outputs,outputs2], name="BB_resnet_2out")
        
       # two losses
        losses = {
        "category_output": "categorical_crossentropy",
        "distance_output": "mean_absolute_error",
        }
        lossWeights = {"category_output": 1.0, "distance_output": 1.0}

        opt = Adam(learning_rate = 0.001)
        
        model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])

        return model
    
    else:
        raise Exception('Model number not recognized')
        
        
    
    