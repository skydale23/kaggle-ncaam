class ModelPrediction:

    def __init__(self):
        self.trained_model=None

    def grid_search_get_best_params(self, X_train, y_train):

        clf = LogisticRegression(random_state = 0)
        params = {'C': np.logspace(start=-5, stop=3, num=50), 'penalty': ['l2']}
        clf = GridSearchCV(clf, params, scoring='neg_brier_score', refit=True)
        X_train_scaled = StandardScaler().fit_transform(X_train)
        clf.fit(X_train_scaled, y_train)
        best_params = clf.best_params_
        return best_params 

    def fit(self, X_train, y_train, params=None):
        ''' train model on unseen data '''

        parameters_to_use = None

        if params is not None: 
            self.params_to_use = params
        
        else:
            self.params_to_use = self.grid_search_get_best_params(X_train, y_train)
        
        lr = Pipeline([('scale', StandardScaler()),('logreg', LogisticRegression(**self.params_to_use))])
        lr.fit(X_train, y_train)
        
        self.trained_model = lr

      
    def predict_proba(self, X_test):
        ''' predict dataset using trained model'''
        preds = self.trained_model.predict_proba(X_test)
        return preds

class ModelValidation:

    def __init__(self, tourney_games, available_features, min_training_size=5, select_features=True):
        self.tourney_games=tourney_games
        self.available_features=available_features
        self.select_features=select_features
        self.min_training_size = min_training_size

    def grid_search_get_best_params(self, X_train, y_train):

        clf = LogisticRegression(random_state = 0)
        params = {'C': np.logspace(start=-5, stop=3, num=50), 'penalty': ['l2']}
        clf = GridSearchCV(clf, params, scoring='neg_brier_score', refit=True)
        X_train_scaled = StandardScaler().fit_transform(X_train)
        clf.fit(X_train_scaled, y_train)
        best_params = clf.best_params_
        return best_params 


    def hyperopt_get_best_params(self, X_train, y_train):

        hp_search_space = {'C': hp.uniform('C', .0001, 1000)}
        oms = OptimizedModelSelector(
                              n_rounds=5
                            , X=X_train
                            , y=y_train
                            , starting_model=LogisticRegression
                            , starting_features=self.available_features
                            , hyper_parameter_space=hp_search_space)
        oms.run()

        return oms.features, oms.selected_params
                
    def validate(self):
        ''' returns estimate for model performance using shifted validation'''

        scores = []
        tourney_games = self.tourney_games
        self.fold_preds = []
        
        for n, season in enumerate(tourney_games.Season.unique()):
            
            if n >= self.min_training_size and n < len(tourney_games.Season.unique()):

                print(season)

                train = tourney_games[tourney_games.Season < season]
                test = tourney_games[tourney_games.Season == season]

                X_train = train[self.available_features]
                X_test = test[self.available_features]
                y_train = train['Outcome'].astype(int)
                y_test = test['Outcome'].astype(int)

                if self.select_features:
                    features, params = self.hyperopt_get_best_params(X_train, y_train)
                    print(features)

                else:
                    features=self.available_features
                    params=self.grid_search_get_best_params(X_train, y_train)

                lr = Pipeline([('scale', StandardScaler()),('logreg', LogisticRegression(**params))])
                lr.fit(X_train[features], y_train)

                y_prob = lr.predict_proba(X_test[features])
                self.fold_preds.append(y_prob)
                loss = brier_score_loss(y_test, y_prob[:,1])
                print(loss)
                scores.append((season, loss))

        self.validation_df = pd.DataFrame(scores, columns = ['season', 'score']).sort_values(by = 'score')
        self.avg_validation_score = self.validation_df.score.mean()


class OptimizedModelSelector:

    def __init__(self, n_rounds, X, y, starting_model, starting_features, hyper_parameter_space, cv_folds=5, default_feature='seed_diff'):
        self.starting_model=starting_model
        self.starting_features=starting_features
        self.hyper_parameter_space=hyper_parameter_space
        self.X = X.copy()
        self.y = y.copy()
        self.folds=cv_folds
        self.n_rounds=n_rounds
        self.default_feature='seed_diff' # feature to use in case model selector tries to be stupid and use 0 features
        self.round_metrics = {} # key is the loss, value is a dict of features and hyperparams

    def objective(self, params):

        selected_features = []
        param_dict = {}

        for i, j in params.items():
            if ('_var' in i) and (j == 1):
                selected_features.append(i)
            elif ('_var' not in i):
                param_dict[i] = j + .000000000001 # avoid divide by zero

        if len(selected_features) == 0:
            selected_features = [self.default_feature + '_var']

        model = Pipeline([('scale', StandardScaler()), ('model', self.starting_model(**param_dict, random_state=0,
                                                                                max_iter=1000))])
       
        cv_scores = cross_val_score(model, self.X[selected_features], self.y, scoring='neg_brier_score', cv=self.folds,
                                error_score='raise', verbose=0)

        return -1 * (sum(cv_scores) / len(cv_scores))
    
    def get_features_and_params_from_best(self, best):

        self.features = []
        self.selected_params = {}
        
        for k,v in best.items():
            if v == 1 and k not in self.hyper_parameter_space.keys():
                self.features.append(k)
            if k in self.hyper_parameter_space.keys():
                self.selected_params[k] = v

    def run(self):

        search_space = self.hyper_parameter_space
        self.features = self.starting_features

        for round in range(self.n_rounds):

            search_space = {}
            
            for k,v in self.hyper_parameter_space.items():
                search_space[k] = v
    
            for col in self.features:
                self.X.rename(columns={col:col+'_var'}, inplace=True)
                search_space[col + '_var'] = hp.choice(col, [0,1])
            
            trials = Trials()

            # print(search_space)

            best = fmin(self.objective, search_space, algo=tpe.suggest,
                                max_evals=100, trials=trials)

            best_loss = trials.best_trial['result']['loss']
            self.get_features_and_params_from_best(best)

            self.round_metrics[best_loss] = {'features': self.features,
                                            'params': self.selected_params}

        # find best score across all rounds, set features / parameters associated with round
        best_score = min(self.round_metrics.keys())
        self.features = self.round_metrics[best_score]['features']
        self.selected_params = self.round_metrics[best_score]['params']


