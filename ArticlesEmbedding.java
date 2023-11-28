package uob.oop;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Properties;


public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";

    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        //TODO Task 5.1 - 1 Mark
        super(_title,_content,_type,_label);
    }

    public void setEmbeddingSize(int _size) {
        //TODO Task 5.2 - 0.5 Marks
        this.intSize = _size;
    }

    public int getEmbeddingSize(){
        return intSize;
    }

    @Override
    public String getNewsContent() {
        //TODO Task 5.3 - 10 Marks
        if(!processedText.isEmpty()){
            return processedText;
        }
        String cleaned = textCleaning(super.getNewsContent());
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,pos,lemma");
        CoreDocument document =  new StanfordCoreNLP(props).processToCoreDocument(cleaned);
        StringBuilder sb = new StringBuilder();
        for(CoreLabel tok : document.tokens()){
            boolean check = false;
            for(String stopWord : Toolkit.STOPWORDS){
                if(tok.lemma().equals(stopWord)){
                    check = true;
                    break;
                }
            }
            if(!check){
                sb.append(tok.lemma().toLowerCase()+" ");
            }

        }
        processedText = sb.toString();

        return processedText.trim();
    } // needs to be reduced in terms of runtime

    public INDArray getEmbedding() throws Exception {
        //TODO Task 5.4 - 20 Marks
        if(intSize==-1){
            throw new InvalidSizeException("Invalid Size");
        }
        if(processedText.isEmpty()){
            throw new InvalidTextException("Invalid text");
        }

        if(!newsEmbedding.isEmpty()) return newsEmbedding;
        String[] words = processedText.split(" "); // words in this article
        int pointer = 0;
        int wordVectorSize = AdvancedNewsClassifier.listGlove.get(0).getVector().getVectorSize();
        double[][] temp = new double[this.intSize][wordVectorSize];

        newsEmbedding = Nd4j.create(intSize,wordVectorSize);


        for (String word : words) {
            boolean isInGlove = false;
            for (Glove x : AdvancedNewsClassifier.listGlove) {
                String currentVocab = x.getVocabulary();

                if (currentVocab.equals(word)) {
                    newsEmbedding.putRow(pointer++, Nd4j.create(x.getVector().getAllElements()));
                    isInGlove = true;
                    break;
                }
            }
            if (pointer >= intSize) {
                break;
            }

        }

        if(pointer<intSize){
            while(pointer<intSize){
                newsEmbedding.putRow(pointer++,Nd4j.create(new double[wordVectorSize]));
            }
        }

        return Nd4j.vstack(newsEmbedding.mean(1));
    }

    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }
}
