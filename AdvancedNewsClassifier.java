package uob.oop;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();
        //TODO Task 6.1 - 5 Marks

        //Toolkit.listVocabulary

        for(int i = 0; i< Toolkit.listVocabulary.size(); i++){
            //for each make a new glove by getting the vector that specific vocab.
            if(!isStopWord(Toolkit.listVocabulary.get(i))){
                Glove tempGlove = new Glove(Toolkit.listVocabulary.get(i), new Vector(Toolkit.listVectors.get(i)));
                listResult.add(tempGlove);
            }
        }
        return listResult;
    }

    public boolean isStopWord(String word){
        for(String stopWord : Toolkit.STOPWORDS){
            if(word.equals(stopWord)){
                return true;
            }
        }

        return false;
    } // Own method that checks if the word is a stopword


    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian = -1;
        //TODO Task 6.2 - 5 Marks
        int[] arr = new int[_listEmbedding.size()];// doc lengths
        StringBuilder sb = new StringBuilder();

        for(Glove glove : listGlove){
            sb.append(glove.getVocabulary()).append(" ");
        }

        int pointer = 0;
        for(ArticlesEmbedding article : _listEmbedding){
            int a = 0;
            for(String x : article.getNewsContent().split(" ")){

                if(sb.indexOf(" "+x+" ")!=-1) a++;
            }
            arr[pointer++] = a;
        } // counts gloves vs article.
              //Selection sort implementation
        for(int i = 0; i<arr.length-1; i++){
            int index = i;
            for (int j = i+1; j < arr.length; j++) {
                if(arr[index]>arr[j]){
                    index = j;
                }
            }
            int temp = arr[index];
            arr[index] = arr[i];
            arr[i] = temp;
        }

        double x = (double) (arr.length + 1) / 2 ;
        if(arr.length % 2 == 0){

            intMedian = (arr[arr.length/2] + arr[(int) Math.round(x)])/2;
        }
        else{

            intMedian = arr[(int) Math.round(x)];
        }


        return intMedian;
    }

    public void populateEmbedding() {
        //TODO Task 6.3 - 10 Marks


        for(ArticlesEmbedding article : listEmbedding){
            try {
                article.getEmbedding();
            } catch (InvalidSizeException f){
                article.setEmbeddingSize(this.calculateEmbeddingSize(listEmbedding));
            } catch (InvalidTextException g){
                article.getNewsContent();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        ListDataSetIterator myDataIterator = null;
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;

        //TODO Task 6.4 - 8 Marks

        return new ListDataSetIterator(listDS, BATCHSIZE);
    }

    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        //TODO Task 6.5 - 8 Marks


        return listResult;
    }

    public void printResults() {
        //TODO Task 6.6 - 6.5 Marks
        
    }
}
