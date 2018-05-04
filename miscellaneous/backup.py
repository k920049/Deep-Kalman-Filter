import tensorflow as tf

x = tf.random_normal(shape=[10, 32, 32, 3])
y = tf.random_normal(shape=[5, 32, 32, 3])

with tf.variable_scope(name_or_scope="convolution", reuse=tf.AUTO_REUSE):
    conv1 = tf.layers.conv2d(inputs=x, filters=3, kernel_size=[3, 3], padding='same', name='conv')
    conv2 = tf.layers.conv2d(inputs=y, filters=3, kernel_size=[3, 3], padding='same', name='conv')

with tf.name_scope("miscellaneous"):
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print([x.name for x in tf.trainable_variables(scope="convolution")])

    param_stack = self.proposal.get_latent_samples(status=status, input=input)
    gen_stack = []
    # variable scope that contains all the weights
    with tf.variable_scope(name_or_scope=self.scope, reuse=tf.AUTO_REUSE):
        top = True
        level = 0
        # at level 0
        with tf.variable_scope(name_or_scope="level" + str(level), reuse=tf.AUTO_REUSE):
            [mu, covariace] = param_stack.pop()
            Q_dist = MultivariateNormalFullCovariance(loc=mu, covariance_matrix=covariace)
            # Q(Z_{t} | Z_{t - 1}
            samples = Q_dist.sample(sample_shape=self.size)
            Q_prob = Q_dist.prob(value=samples)
            # covariance of P(Z_{t} | Z_{t - 1}, U_{t - 1})
            with tf.variable_scope(name_or_scope="covariance", reuse=tf.AUTO_REUSE):
                G = tf.get_variable(name="G", shape=(self.num_units, self.num_units), dtype=tf.float32)
                Z_samples = tf.matmul(samples, G)
                prev_covariance = tf.matmul(G, tf.matmul(covariace, tf.transpose(G)))

            # mean of P(X_{t} | Z_{t})
            with tf.variable_scope(name_or_scope="X", reuse=tf.AUTO_REUSE):
                mean = tf.zeros(shape=(1, self.num_units))
                X_samples = tf.layers.dense(inputs=Z_covariance,
                                            units=self.num_units - 1,
                                            name="F")
                X_mu = tf.layers.dense(inputs=mean,
                                       units=self.num_units - 1,
                                       name="F")
                X_theta = tf.get_variable(name="theta",
                                          dtype=tf.float32,
                                          initializer=self.init)
                X_dist = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=theta)
                X_prob = X_dist.prob(value=X_samples)
        prev_layer = Z_samples
        prev_mean = tf.zeros(shape=(1, self.num_units))
        P_dist = MultivariateNormalFullCovariance(loc=prev_mean, covariance_matrix=prev_covariance)
        P_prob = P_dist.prob(value=prev_layer)
        level = level + 1

        while len(param_stack) != 0:
            with tf.variable_scope(name_or_scope="level" + str(level), reuse=tf.AUTO_REUSE):
                # get another parameters from the stack
                [mu, covariace] = param_stack.pop()
                Q_dist = MultivariateNormalFullCovariance(loc=mu, covariance_matrix=covariace)
                # Q(Z_{t} | Z_{t - 1}
                samples = Q_distdist.sample(sample_shape=self.size)
                Q_prob = Q_dist.prob(value=samples)
                # covariance of P(Z_{t} | Z_{t - 1}, U_{t - 1})
                with tf.variable_scope(name_or_scope="covariance", reuse=tf.AUTO_REUSE):
                    G = tf.get_variable(name="G",
                                        shape=(self.num_units, self.num_units),
                                        dtype=tf.float32)
                    Z_samples = tf.matmul(samples, G)
                    prev_covariance = tf.matmul(G, tf.matmul(covariace, tf.transpose(G)))

                with tf.variable_scope(name_or_scope="forward", reuse=tf.AUTO_REUSE):
                    cur_layer = prev_layer
                    for l in range(self.num_layers):
                        cur_layer = tf.layers.dense(inputs=cur_layer,
                                                    units=self.num_units,
                                                    use_bias=False
                        name = "G_layer" + str(l))

                        for l in range(self.num_layers):
                            prev_mean = tf.layers.dense(inputs=prev_mean,
                                                        units=self.num_units,
                                                        use_bias=False
                            name = "G_layer" + str(l))
                            cur_layer = cur_layer + Z_covariance
                            P_dist = MultivariateNormalFullCovariance(loc=prev_mean, covariance_matrix=prev_covariance)
                            P_prob = P_dist.prob(value=cur_layer)
                            # mean of P(X_{t} | Z_{t})
                            with tf.variable_scope(name_or_scope="X", reuse=tf.AUTO_REUSE):
                                X_samples = tf.layers.dense(inputs=cur_layer,
                                                            units=self.num_units - 1,
                                                            use_bias=False,
                                                            name="F")
                                X_mu = tf.layers.dense(inputs=prev_mean,
                                                       units=self.num_units - 1,
                                                       use_bias=False,
                                                       name="F")
                                X_theta = tf.get_variable(name="theta",
                                                          dtype=tf.float32,
                                                          initializer=self.init)
                                X_dist = MultivariateNormalFullCovariance(loc=mean, covariance_matrix=theta)
                                X_prob = X_dist.prob(value=X_samples)

                        prev_layer = cur_layer
                        level = level + 1