package uob.oop;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public class Toolkit {
    public static List<String> listVocabulary = null;
    public static List<double[]> listVectors = null;
    private static final String FILENAME_GLOVE = "glove.6B.50d_Reduced.csv";

    public static final String[] STOPWORDS = {"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"};

    public void loadGlove() throws IOException {
        BufferedReader myReader = null;
        //TODO Task 4.1 - 5 mark
        listVocabulary = new ArrayList<>();
        listVectors = new ArrayList<>();
        try {
            File file = Toolkit.getFileFromResource(FILENAME_GLOVE);
            FileReader fr = new FileReader(file);
            myReader = new BufferedReader(fr);
            String i;
            while((i=myReader.readLine())!=null){
                String[] temp = i.split(" ");
                for(String x : temp){
                    listVocabulary.add(x.substring(0,x.indexOf(",")));
                    String[] val = x.substring(x.indexOf(",")+1).split(",");
                    double[] doubles = new double[val.length];
                    int pointer = 0;
                    for(String str : val){
                        doubles[pointer] = Double.parseDouble(val[pointer++]);
                    }
                    listVectors.add((doubles));
                }
            }
            myReader.close();
        } catch (URISyntaxException e) {
            System.out.println(e.getMessage());
            throw new RuntimeException(e);
        }


    }

    private static File getFileFromResource(String fileName) throws URISyntaxException {
        ClassLoader classLoader = Toolkit.class.getClassLoader();
        URL resource = classLoader.getResource(fileName);
        if (resource == null) {
            throw new IllegalArgumentException(fileName);
        } else {
            return new File(resource.toURI());
        }
    }

    public List<NewsArticles> loadNews() {
        List<NewsArticles> listNews = new ArrayList<>();
        //TODO Task 4.2 - 5 Marks

        File dir = new File("src/main/resources/News"); // ask jizeng
        File[] fileList = dir.listFiles();

        // sort files here.

        if(fileList != null){
            for (int i = 0; i < fileList.length - 1; i++) {
                for (int j = 0; j < fileList.length - i - 1; j++) {
                    if (fileList[j].getName().compareTo(fileList[j + 1].getName()) > 0) {
                        File temp = fileList[j];
                        fileList[j] = fileList[j + 1];
                        fileList[j + 1] = temp;
                    }
                }
            }
            for(File file : fileList){
                if(file.getName().endsWith(".htm")){
                    StringBuilder sb = new StringBuilder();

                    try(BufferedReader br = Files.newBufferedReader(file.toPath())){
                        String line;
                        while((line = br.readLine()) != null){
                            sb.append(line).append("\n");
                        }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    String x = sb.toString();
                    String title = HtmlParser.getNewsTitle(x);
                    String content =  HtmlParser.getNewsContent(x);
                    NewsArticles.DataType type = HtmlParser.getDataType(x);
                    String label = HtmlParser.getLabel(x);
                    listNews.add(new NewsArticles(title, content, type, label));
                    // load html code into sb
                    //gettitle, content, datatype and label.
                    // make a newsarticle object and initialise w that.
                    // add to listNewsVariable
                }
            }
        }

        return listNews;
    }


    public static List<String> getListVocabulary() {
        return listVocabulary;
    }

    public static List<double[]> getlistVectors() {
        return listVectors;
    }
}
