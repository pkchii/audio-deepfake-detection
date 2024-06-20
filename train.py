import models.build_model as BM
import tensorflow.keras as keras


if __name__ == "__main__":

    
    X_train, X_validation, X_test, y_train, y_validation, y_test = BM.prepare_datasets(0.25, 0.2)

    
    input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13
    model = BM.build_lstm(input_shape)

    # input_shape = (130, 13, 1)
    # model = BM.build_cnn(input_shape)

    
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    
    BM.plot_history(history)

    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    
    X_to_predict = X_test[100]
    y_to_predict = y_test[100]

    
    BM.predict(model, X_to_predict, y_to_predict)