"""
       batch_gen = get_batch(X_train)

       while True:
           index = index + 1
           try:
               batch = next(batch_gen)
           except StopIteration:
               epoch = epoch + 1
               batch_gen = get_batch(X_train)
               continue

           loss, _ = sess.run([loss, ops], feed_dict={X: batch})
           print("At epoch {} iteration {}, loss -> {}".format(epoch, index, loss))
       """

