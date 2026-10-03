%% Описание изображений простыми дескрипторами и распознавание объектов на основе сравнения с эталоном
% Наборы данных:
% - изображения букв английского алфавита notMNIST (kaggle.com);
% - изображения одежды Fashion MNIST PNG (kaggle.com).

%% 1. Загрузка изображений и вычисление дескрипторов

% Получение доступа к файлам тренировочных изображений (закомментировать для тестирования)
imdDir = imageDatastore('C:\Users\gps.84\Downloads\archive\test',"IncludeSubfolders",true,"LabelSource","foldernames");

% Получение доступа к файлам тестовых изображений (раскомментировать для тестирования)
% imdDir = imageDatastore('C:\Users\gps.84\Downloads\archive\test',"IncludeSubfolders",true,"LabelSource","foldernames");

% Инициализация массивов для дескрипторов
D_1=zeros(length(imdDir.Files),2*28,'single');
D_2=zeros(length(imdDir.Files),28,'single');
D_3=D_2;
D_4=zeros(length(imdDir.Files),4,'single');
label =categorical([]);

% Вычисление дескрипторов для каждого изображения
tic
parfor i=1:length(imdDir.Files) 
    try

    % Загрузка изображения из файла    
    I=imread(imdDir.Files{i,1}); 
    
    % Пороговая бинаризация изображения
    I = imbinarize(I,'global'); 

    % Вычисление дескрипторов
    X=sum(I,1); % сумма значений пикселей по столбцам
    Y=sum(I,2); % сумма значений пикселей по строкам
    D_1(i,:)=[X Y']; % вариант дескриптора № 1
    D_2(i,:)=X-Y'; % вариант дескриптора № 2
    D_3(i,:)=abs(X-Y'); % вариант дескриптора № 3
    D_4(i,:)=[mean(D_1(i,:)) std(D_1(i,:)) mode(D_1(i,:)) median(D_1(i,:))]; % вариант дескриптора № 4
    
    % Метки классов набора
    label(i,1)=imdDir.Labels(i,1); 

    catch
        disp('Ошибка чтения файла!')
    end
end 
toc
%% Вычисление центров кластеров дескрипторов обучающего набора данных

% Уникальные метки кластеров
unique_label = unique(label)';

% Для каждого класса объекта
for i = unique_label
    
    % Логическая маска для поиска элементов соответствующих unique_label
    mask = (label == i);

    % Вычисление центров кластеров для различных классов объектов
    cla_D_1(i,:) = mean(D_1(mask,:));
    cla_D_2(i,:) = mean(D_2(mask,:));
    cla_D_3(i,:) = mean(D_3(mask,:));
    cla_D_4(i,:) = mean(D_4(mask,:));
end

%% Распознавание тестовых изображений и оценка точности классификации

% Классификация изображений на основе дескриптора D_1 
predict = predict_images(cla_D_1, D_1, unique_label);
figure(1), confusionchart(label,predict,"Normalization","column-normalized");
Acc = mean(label==predict);
fprintf('Усредненное значение точности классификации изображений на основе D_1: %f \n', Acc)

% Классификация изображений на основе дескриптора D_2 
predict = predict_images(cla_D_2, D_2, unique_label);
figure(2), confusionchart(label,predict,"Normalization","column-normalized");
Acc = mean(label==predict);
fprintf('Усредненное значение точности классификации изображений на основе D_2: %f \n', Acc)

% Классификация изображений на основе дескриптора D_3 
predict = predict_images(cla_D_3, D_3, unique_label);
figure(3), confusionchart(label,predict,"Normalization","column-normalized");
Acc = mean(label==predict);
fprintf('Усредненное значение точности классификации изображений на основе D_3: %f \n', Acc)

% Классификация изображений на основе дескриптора D_4 
predict = predict_images(cla_D_4, D_4, unique_label);
figure(4), confusionchart(label,predict,"Normalization","column-normalized");
Acc = mean(label==predict);
fprintf('Усредненное значение точности классификации изображений на основе D_4: %f \n', Acc)

%% Вспомогательные функции программы

function predict = predict_images(cla_D, D, unique_label)
% Вычисление модуля разности между кластерными центрами и дескрипторами
% тестовых изображений, усреднение по строкам и нахождение кластера до
% которого расстояние минимально
predict = categorical([]);
for j=1:size(D,1)
    [k,l]=min(mean(abs(cla_D - D(j,:)),2));
    predict(j,1) =  unique_label(l);
end
end

