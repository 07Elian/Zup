#Pacotes utilizados

pacotes <- c("plotly","tidyverse","knitr","kableExtra","fastDummies","rgl","car",
             "reshape2","jtools","lmtest","caret","pROC","ROCR","nnet","kknn","dismo","MASS","h2o","kernlab","dummy","mfx")

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T) 
} else {
  sapply(pacotes, require, character = T) 
}
h2o.init(nthreads = -1)

#-------------- BASE E PRE-PROCESSAMENTO BASICO

dados  = read.csv(choose.files())
dados = dados[,-c(9,10,22,27)]
Attrition = as.numeric(ifelse(dados$Attrition=='Yes',1,0))
dados$Attrition = Attrition
head(dados)

# FUNCOES DE PRE-PROCESSAMENTO

normalizacao = function(a){
  aux  = as.numeric(scale(a))
  return(aux)
}

# FUN��ES DA LOGISTICA BIVARIADA SIMPLES

lBScategorica = function(a){
  VE = dummy_columns(.data = a, remove_selected_columns = T,remove_first_dummy = T)
  teste =data.frame(Attrition,VE)
  modelo = glm( Attrition ~ ., family = binomial, data = teste) 
  aux = summ(modelo, confint = T, digits = 5, ci.width = .95) #função summ do pacote jtools
  return(aux)
}

lBSnumerica = function(a){
  VE = a
  teste =data.frame(Attrition,VE)
  modelo = glm( Attrition ~ ., family = binomial, data = teste) 
  aux = summ(modelo, confint = T, digits = 5, ci.width = .95) #função summ do pacote jtools
  return(aux)
}  
  
RCcategorica = function(a){
  VE = dummy_columns(.data = a, remove_selected_columns = T,remove_first_dummy = T)
  teste =data.frame(Attrition,VE)
  aux = logitor( formula= as.formula("Attrition ~ ."), data = teste) 
  return(aux)
}

RCnumerica = function(a){
  VE = a
  teste =data.frame(Attrition,VE)
  aux = logitor( formula= as.formula("Attrition ~ ."), data = teste)
  return(aux)
}  

# FUN��ES DE PREDICAO

AuxSaida = function(pred, teste, formula, alg){
  ind = which(colnames(teste) == as.character(formula)[2])
  rs = Medidas(pred, teste[, ind])
  rs[['Classificador']] = alg    
  return(do.call(data.frame, rs))  
}

Medidas = function(pred, obsv){
  vp = sum((pred == 1) & (obsv == 1)) # mc[2,2] 
  vn = sum((pred == 0) & (obsv == 0)) # mc[1,1]
  fp = sum((pred == 1) & (obsv == 0)) # mc[2,1]
  fn = sum((pred == 0) & (obsv == 1)) # mc[1,2]
  acc = (vp + vn)/(vp + vn + fp + fn)
  sen = vp/(vp + fn) #Recall/Sensibilidade
  esp = vn/(vn + fp) #Especificidade
  vpp = vp/(vp + fp) #Precision/Valor preditivo positivo
  vpn = vn/(vn + fn) #Valor preditivo negativo
  mcc = (vp*vn - fp*fn)/(sqrt((vp + fp)*(vp + fn)*(vn + fp)*(vn + fn))) #Coeficiente de correlacao de Matthews
  mcc = ifelse(is.na(mcc), 0, mcc)
  bacc = (sen + esp)/2
  f1 = (2*sen * esp)/(sen + esp)
  
  return(list(Acc = acc, Sens = sen, Espec = esp, VPP = vpp, VPN = vpn, MCC = mcc, BAcc = bacc,F1=f1))
}

melhorK = function(dados, formula, inicio = 1, fim = 51, metrica = 'Acc', ...){
  knns = data.frame()
  s = seq(inicio, fim, by = 2)
  for(k in s) {
    cat('Calculando para:', k, 'vizinhos\n')
    knns = rbind(knns, ValCruzRep(medKNN, dados, formula, K = k, ...)$Media)
  }
  ordem = order(knns[metrica])
  melhorK = s[tail(ordem, 1)]
  ord = knns[ordem, ]
  par(mai = c(1, 1, 1, 1))
  plot(s, knns[[metrica]], pch = 17, xlab = 'Valores de K (Vizinhos)', main = paste('Melhor K segundo a métrica:', metrica),
       ylab = metrica, xaxt = "n", col = 'dodgerblue4') 
  axis(1, at = s)
  segments(x0 = melhorK, y0 = 0, x1 = melhorK, y1 = as.numeric(tail(ord[metrica], 1)), 
           lty = 'dotted', col = 'gray50', lwd = 2)
  return(melhorK)
}

Todos = function(){
  tds = lsf.str(envir = .GlobalEnv)
  fcs = tds[grepl('med', tds) ] # retorna lista de todas as funcoes carregadas que comecam com 'med'
  aux = substring(fcs, 4)
  return(aux)
}

FormFac = function(formula){
  aux = as.character(formula)[2]
  return(as.formula(paste('factor(',aux, ')', '~ .')))
}


################################
### METODOS DE CLASSIFICACAO ###
################################

RegLog = function(treino, teste, formula, ...){ 
  cls = glm(formula, data = treino, family = 'binomial')
  pred = predict(cls, newdata = teste, type = 'response') 
  pred = ifelse(pred > 0.5, 1, 0) 
  pred = as.numeric(pred)
  return(pred)
}

medRegLog = function(treino, teste, formula, ...){
  alg = 'Regressão Logistica' 
  pred = RegLog(treino, teste, formula)
  return(AuxSaida(pred, teste, formula, alg))
}


Arvore = function(treino, teste, formula, ...){
  library(rpart) 
  cls = rpart(formula, data = treino, method = 'class')
  pred = predict(cls, newdata = teste, type = 'class')
  pred = as.numeric(pred) - 1
  return(pred)
}

medArvore = function(treino, teste, formula, ...){ 
  alg = '�rvore de Decisão' 
  pred = Arvore(treino, teste, formula)
  return(AuxSaida(pred, teste, formula, alg))
}

RF = function(treino, teste, formula, ...){
  library(randomForest)
  RF = randomForest(FormFac(formula), data = treino)
  pred = predict(RF, newdata = teste, type = 'class')
  pred = as.numeric(pred) - 1
  return(pred)
}

medRF = function(treino, teste, formula, ...){ 
  alg = 'Random Forest' 
  pred = RF(treino, teste, formula)
  return(AuxSaida(pred, teste, formula, alg))
}

KNN = function(treino, teste, formula, K = 15, ...){
  library(kknn) 
  cls = kknn(FormFac(formula), train = treino, test = teste, k = K) 
  pred = cls$fitted.values 
  pred = as.numeric(pred) - 1
  return(pred)
}

medKNN = function(treino, teste, formula, K = 15, ...){ 
  alg = paste(K, '-Vizinhos mais Próximos', sep = '')
  pred = KNN(treino, teste, formula, K)
  return(AuxSaida(pred, teste, formula, alg))
}

LDA = function(treino, teste, formula, ...){
  cls = lda(formula, data = treino)
  pred = predict(cls, newdata = teste)$class 
  pred = as.numeric(pred) - 1
  return(pred)
}

medLDA = function(treino, teste, formula, ...){ 
  alg = 'Análise Discriminante Linear' 
  pred = LDA(treino, teste, formula)
  return(AuxSaida(pred, teste, formula, alg))
}

QDA = function(treino, teste, formula, ...){
  library(MASS) 
  cls = qda(formula, data = treino)
  pred = predict(cls, newdata = teste)$class 
  pred = as.numeric(pred) - 1
  return(pred)
}

medQDA = function(treino, teste, formula, ...){
  alg = 'Análise Discriminante Quadrática' 
  pred = QDA(treino, teste, formula) 
  return(AuxSaida(pred, teste, formula, alg))
}

Bagging = function(treino, teste, formula, classif = Arvore, B = 25, ...){
  preds = NULL
  for(i in 1:B){
    indBS = sample(nrow(treino), replace = TRUE)
    repBS = treino[indBS, ]
    pred = classif(treino = repBS, teste = teste, formula)
    preds = cbind(preds, pred)
  }
  votos = rowMeans(preds)
  pred = ifelse(votos > 0.5, 1, 0) 
  return(pred)
}

medBagging = function(treino, teste, formula, classif = Arvore, B = 25, ...){
  alg = paste(B, '-Bagging', sep = '') 
  pred = Bagging(treino, teste, formula, classif, B) 
  return(AuxSaida(pred, teste, formula, alg))
}

ANN = function(treino, teste, formula, camadas = c(5,5), ...){
  classifier = h2o.deeplearning(y = as.character(formula)[2],
                                training_frame = as.h2o(as.matrix(treino)),
                                activation = 'Rectifier',
                                hidden = camadas,
                                epochs = 1000,
                                train_samples_per_iteration = -2)
  prob_pred = h2o.predict(classifier, newdata = as.h2o(as.matrix(teste)))
  pred = (prob_pred > 0.5)
  pred0 = as.vector(pred)
  return(pred0)
}

medANN = function(treino, teste, formula, camadas = c(5,5), ...){
  alg = 'Redes Neurais' 
  pred = ANN(treino, teste, formula, camadas) 
  return(AuxSaida(pred, teste, formula, alg))
}

SVM = function(treino, teste, formula){
  mod <- ksvm(formula,data=treino,
              method="C-svc",
              kernel=besseldot(sigma = 1, order = 2, degree = 1))
  pred <- predict(mod,teste)
  return(pred)
}

medSVM = function(treino, teste, formula, ...){
  alg = 'SVM' 
  pred = SVM(treino, teste, formula) 
  return(AuxSaida(pred, teste, formula, alg))
}

#########################
### VALIDA��O CRUZADA ###
#########################

ValidCruzada = function(cls, dados, formula, kfolds = 10, ...){
  library(dismo)
  id = kfold(1:nrow(dados), kfolds)
  vcs = data.frame()
  for(i in 1:kfolds){
    treino = dados[id != i, ]
    teste = dados[id == i, ]
    kcls = cls(treino, teste, formula, ...) 
    vcs = rbind(vcs, kcls)
  }
  vcs = vcs[, c(ncol(vcs), 1:(ncol(vcs) - 1))]
  avg = aggregate(. ~ Classificador, data = vcs, FUN = mean)
  std = aggregate(. ~ Classificador, data = vcs, FUN = sd)
  return(list(Media = avg, Desvio = std, Modelos = vcs))
}


##################################
### VALIDACAO CRUZADA REPETIDA ###
##################################

ValCruzRep = function(cls, dados, formula, kfolds = 10, reps = 10, ...){
  x = data.frame()
  for(i in 1:reps) x = rbind(x, ValidCruzada(cls, dados, formula, kfolds, ...)$Media) 
  avg = aggregate(. ~ Classificador, data = x, FUN = mean)
  std = aggregate(. ~ Classificador, data = x, FUN = sd)
  return(list(Media = avg, Desvio = std, Modelos = x))
}

TodosClass = function(cls = Todos(), dados, formula, metrica = 'Acc', ...){
  fcls = paste('med', cls, sep = '')
  x = y = data.frame()
  for(i in 1:length(cls)){ 
    classif = get(fcls[[i]])
    cat('Calculando classificador: ', i, '/', length(cls), sep = '', end = '\n')
    aux = ValCruzRep(classif, dados, formula, ...)
    x = rbind(x, aux$Modelos)
    y = rbind(y, aux$Media) 
  } 
  cat('\nPronto\n')
  y = y[rev(order(y[metrica])), ]
  par(mai = c(1, 3, 1, 1))
  cores = colorRampPalette(c('cadetblue3', 'darkcyan'))
  barplot(rev(y[[metrica]]), main = 'Performance dos Classificadores', col = cores(length(y[[metrica]])),
          horiz = TRUE, names.arg = rev(y[['Classificador']]), las = 1, xlim = c(0, 1), xlab = metrica)
  abline(v = y[[metrica]][1], col = 'gray50', lty = 'dashed')
  return(list(Modelos = x, Media = y ))
}


# ------------- VANALISE BIVARIADA - SIMPLES

lBSnumerica(dados$Age)
lBSnumerica(dados$DailyRate)
lBSnumerica(dados$DistanceFromHome)
lBSnumerica(dados$HourlyRate) # não e significativa
lBSnumerica(dados$MonthlyIncome) 
lBSnumerica(dados$MonthlyRate)
lBSnumerica(dados$NumCompaniesWorked) # não e significativa
lBSnumerica(dados$PercentSalaryHike) # não e significativa
lBSnumerica(dados$TotalWorkingYears)
lBSnumerica(dados$TrainingTimesLastYear)
lBSnumerica(dados$YearsAtCompany)
lBSnumerica(dados$YearsInCurrentRole)
lBSnumerica(dados$YearsSinceLastPromotion) # não e significativa
lBSnumerica(dados$YearsWithCurrManager)

lBScategorica(dados$BusinessTravel)
lBScategorica(dados$Department) # não e significativa
lBScategorica(dados$Education) # não e significativa
lBScategorica(dados$EducationField) # não e significativa
lBScategorica(dados$EnvironmentSatisfaction) 
lBScategorica(dados$Gender) # não e significativa
lBScategorica(dados$JobInvolvement) 
lBScategorica(dados$JobLevel)
lBScategorica(dados$JobRole) # parcialmente
lBScategorica(dados$JobSatisfaction)
lBScategorica(dados$MaritalStatus)# parcialmente
lBScategorica(dados$OverTime)
lBScategorica(dados$PerformanceRating)  # não e significativa
lBScategorica(dados$RelationshipSatisfaction) # parcialmente
lBScategorica(dados$StockOptionLevel)
lBScategorica(dados$WorkLifeBalance)

#----------------------------------- MODELO LOGISTICO COM TODAS AS VARIAVEIS

#dumies
baseModeloLogistico = dummy_columns(.data = dados,
              select_columns = c("BusinessTravel", 
                                 "Department",
                                 "Education", 
                                 "EducationField",
                                 "EnvironmentSatisfaction",
                                 "Gender",
                                 "JobInvolvement",
                                 "JobLevel",
                                 "JobRole",
                                 "JobSatisfaction",
                                 "MaritalStatus",
                                 "OverTime",
                                 "PerformanceRating",
                                 "RelationshipSatisfaction",
                                 "StockOptionLevel",
                                 "WorkLifeBalance"),
              remove_selected_columns = T,
              remove_first_dummy = T)

modeloInferencia = glm( Attrition ~ ., family = binomial, data = baseModeloLogistico) 
summ(modeloPredicao, confint = T, digits = 3, ci.width = .95) 

step_modeloInferencia<- step(object = modeloInferencia,
                            k = qchisq(p = 0.05, df = 1, lower.tail = FALSE))

formulaInferencia = as.formula("Attrition ~ Age + DistanceFromHome + MonthlyIncome + NumCompaniesWorked + 
    TotalWorkingYears + TrainingTimesLastYear + YearsAtCompany + 
    YearsInCurrentRole + YearsSinceLastPromotion + YearsWithCurrManager + 
    BusinessTravel_Travel_Frequently + BusinessTravel_Travel_Rarely + 
    `EducationField_Life Sciences` + EducationField_Medical + 
    EducationField_Other + EnvironmentSatisfaction_2 + EnvironmentSatisfaction_3 + 
    EnvironmentSatisfaction_4 + Gender_Male + JobInvolvement_2 + 
    JobInvolvement_3 + JobInvolvement_4 + JobLevel_2 + JobLevel_5 + 
    `JobRole_Research Scientist` + `JobRole_Sales Executive` + 
    JobSatisfaction_2 + JobSatisfaction_3 + JobSatisfaction_4 + 
    MaritalStatus_Single + OverTime_Yes + RelationshipSatisfaction_2 + 
    RelationshipSatisfaction_3 + RelationshipSatisfaction_4 + 
    StockOptionLevel_1 + StockOptionLevel_2 + WorkLifeBalance_2 + 
    WorkLifeBalance_3 + WorkLifeBalance_4")

summ(step_modeloInferencia, confint = T, digits = 3, ci.width = .95) 

razaoChancesInferencia = logitor(formula = formulaInferencia, data = baseModeloLogistico)


# -----------------------------PREDICAO-----------------------------------------------

# Normaliza��o das variaveis quantitativas
basePredicao = dados
basePredicao$Age = normalizacao(basePredicao$Age)
basePredicao$DailyRate = normalizacao(basePredicao$DailyRate)
basePredicao$DistanceFromHome = normalizacao(basePredicao$DistanceFromHome)
basePredicao$HourlyRate = normalizacao(basePredicao$HourlyRate )
basePredicao$MonthlyIncome = normalizacao(basePredicao$MonthlyIncome )
basePredicao$MonthlyRate = normalizacao(basePredicao$MonthlyRate)
basePredicao$NumCompaniesWorked = normalizacao(basePredicao$NumCompaniesWorked)
basePredicao$PercentSalaryHike = normalizacao(basePredicao$PercentSalaryHike)
basePredicao$TotalWorkingYears = normalizacao(basePredicao$TotalWorkingYears)
basePredicao$TrainingTimesLastYear = normalizacao(basePredicao$TrainingTimesLastYear)
basePredicao$YearsAtCompany = normalizacao(basePredicao$YearsAtCompany )
basePredicao$YearsInCurrentRole = normalizacao(basePredicao$YearsInCurrentRole)
basePredicao$YearsSinceLastPromotion = normalizacao(basePredicao$YearsSinceLastPromotion)
basePredicao$YearsWithCurrManager = normalizacao(basePredicao$YearsWithCurrManager)

# transforma�ao das variaveis qualitativas em dummies

basePredicao = dummy_columns(.data = basePredicao,
                                    select_columns = c("BusinessTravel", 
                                                       "Department",
                                                       "Education", 
                                                       "EducationField",
                                                       "EnvironmentSatisfaction",
                                                       "Gender",
                                                       "JobInvolvement",
                                                       "JobLevel",
                                                       "JobRole",
                                                       "JobSatisfaction",
                                                       "MaritalStatus",
                                                       "OverTime",
                                                       "PerformanceRating",
                                                       "RelationshipSatisfaction",
                                                       "StockOptionLevel",
                                                       "WorkLifeBalance"),
                                    remove_selected_columns = T,
                                    remove_first_dummy = T)

modeloPredicao = glm( Attrition ~ ., family = binomial, data = basePredicao) 
summ(modeloPredicao, confint = T, digits = 3, ci.width = .95) 

step_modeloPredicao <- step(object = modeloPredicao,
                        k = qchisq(p = 0.05, df = 1, lower.tail = FALSE))
summ(step_modeloPredicao, confint = T, digits = 3, ci.width = .95) 

fomulaPredicao = as.formula("Attrition ~ Age + DistanceFromHome + MonthlyIncome + NumCompaniesWorked + 
    TotalWorkingYears + TrainingTimesLastYear + YearsAtCompany + 
    YearsInCurrentRole + YearsSinceLastPromotion + YearsWithCurrManager + 
    BusinessTravel_Travel_Frequently + BusinessTravel_Travel_Rarely + 
    `EducationField_Life Sciences` + EducationField_Medical + 
    EducationField_Other + EnvironmentSatisfaction_2 + EnvironmentSatisfaction_3 + 
    EnvironmentSatisfaction_4 + Gender_Male + JobInvolvement_2 + 
    JobInvolvement_3 + JobInvolvement_4 + JobLevel_2 + JobLevel_5 + 
    `JobRole_Research Scientist` + `JobRole_Sales Executive` + 
    JobSatisfaction_2 + JobSatisfaction_3 + JobSatisfaction_4 + 
    MaritalStatus_Single + OverTime_Yes + RelationshipSatisfaction_2 + 
    RelationshipSatisfaction_3 + RelationshipSatisfaction_4 + 
    StockOptionLevel_1 + StockOptionLevel_2 + WorkLifeBalance_2 + 
    WorkLifeBalance_3 + WorkLifeBalance_4")

melhorK(basePredicao, fomulaPredicao, 1 , 21, 'F1', kfolds = 10, reps = 5)

TodosClass(cls = c('QDA','RegLog','Arvore','KNN','LDA','ANN'), 
           dados = basePredicao, formula = fomulaPredicao, metrica = 'F1',
           reps = 10, kfolds = 10, K = 3, B = 50)






