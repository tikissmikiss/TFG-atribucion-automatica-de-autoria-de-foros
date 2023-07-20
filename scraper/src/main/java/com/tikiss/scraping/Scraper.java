package com.tikiss.scraping;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.net.Proxy;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.tikiss.scraping.entity.Cita;
import com.tikiss.scraping.entity.Post;
import org.jsoup.Connection;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import lombok.extern.java.Log;

@Log
public class Scraper implements Runnable {

    private static final int PROXY_PORT = 8080;
    private static final String PROXY_IP = "127.0.0.1";

    private String urlBase = "https://forocoches.com/foro/showthread.php";
    private String extension = ".json";
    private String folder = "./json/v3/";

    private String fullUrl;
    private String nameHilo = null;
    private final List<Post> listPosts = new ArrayList<>();
    private int iniPag = 1;
    private int endPag = 1;
    private int idThread;
    private int iniThread = 15505;
    private int endThread = 9999999;

    public Scraper() {
    }

    public Scraper(int idThread) {
        this.iniThread = idThread;
        this.endThread = idThread;
    }

    public Scraper(int iniThread, int endThread) {
        this.iniThread = iniThread;
        this.endThread = endThread;
    }

    public Scraper(String urlBase, int idThread) {
        this.urlBase = urlBase;
        this.idThread = idThread;
    }

    public Scraper(String fullUrl, String fileName, String folder) {
        this.fullUrl = fullUrl;
        this.extension = fileName;
        this.folder = folder;
    }

    public Scraper(String fullUrl, int pages, String atrbPage, String fileName, String folder) {
        this.fullUrl = fullUrl;
        // this.urlBase = url + "&" + atrbPage + "=";
        this.endPag = pages;
        this.extension = fileName;
        this.folder = folder;
    }

    public Scraper(String fullUrl, int iniPag, int endPag, String atrbPage, String fileName, String folder) {
        this.fullUrl = fullUrl;
        // this.urlBase = url + "&" + atrbPage + "=";
        this.iniPag = iniPag;
        this.endPag = endPag;
        this.extension = fileName;
        this.folder = folder;
    }

    public void run() {
        // log.log(Level.INFO, "Running");

        InetSocketAddress socket = new InetSocketAddress(PROXY_IP, PROXY_PORT);
        Proxy proxy = new Proxy(Proxy.Type.HTTP, socket);
        Proxy socksProxy = new Proxy(Proxy.Type.SOCKS, socket);

        for (int idThread = iniThread; idThread <= endThread; idThread++) {
            boolean isThreadEmpty = true;
            endPag = 1;

            for (int pag = iniPag; pag <= endPag; pag++) {
                fullUrl = buildFullUrl(idThread, pag);
                
                // fullUrl = urlBase + "&page=" + pag;
                Connection connection = Jsoup.connect(fullUrl)
                .userAgent("Mozilla")
                .header("X-Forwarded-For", "1.2.3.4")
                // .proxy(PROXY_IP, PROXY_PORT)
                .proxy(proxy)
                .validateTLSCertificates(false);
                
                Document doc;
                try {
                    doc = connection.get();
                    String msg = "";
                    Elements h1 = doc.getElementsByTag("h1");
                    isThreadEmpty = h1.isEmpty();
                    if (isThreadEmpty) 
                        msg = "Acceso Restringido";
                        
                    if (!isThreadEmpty) {
                        this.nameHilo = h1.first().text();
                        isThreadEmpty = nameHilo.equals("Información");
                        if (isThreadEmpty) 
                            msg = "No existe el hilo. Ningún Tema especificado.";
                    }
                    if (!isThreadEmpty) {
                        if (endPag == 1)
                            endPag = getNumPags(doc);
                        isThreadEmpty = endPag < 50;
                        if (isThreadEmpty) {
                            endPag = 1;
                            msg = "Hilo con menos de 50 paginas - " + doc.getElementsByTag("section").get(1).text();
                        }
                    }
                    if (!isThreadEmpty) {
                        System.out.println(Thread.currentThread().getName() + ": Hilo " + idThread + " - Pagina " + pag + " de " + endPag + ": " + nameHilo);
                        
                        Elements postElements = doc.getElementsByClass("postbit_wrapper");

                        for (Element element : postElements) {
                            try {
                                Post post = extractPost(element);
                                if (post.getContenido().length() < 10)
                                    throw new Exception("Contenido bajo: " + post.getContenido());
                                listPosts.add(post);
                            } catch (Exception e) {
                                System.out.println(Thread.currentThread().getName() + ": :" + idThread + ": Post ignorado por error: " + e.getMessage());
                            }
                        }
                    } else {
                        // System.out.println(Thread.currentThread().getName() + ": Hilo ignorado " + idThread + " - " + msg);
                    }
                } catch (IOException e1) {
                    System.out.println(Thread.currentThread().getName() + ": ERROR IOException en idThread " + idThread);
                }
            }
            
            if (!isThreadEmpty) {
                ObjectMapper mapper = new ObjectMapper();
                String json;
                try {
                    json = mapper.writeValueAsString(listPosts);

                    String file = String.format("%05d posts - %s [%d-%d]%s",
                            listPosts.size(), nameHilo, iniPag, endPag, extension);
                    // String file = nameHilo + "[" + iniPag + "-" + endPag + "]";
                    // file += " - " + listPosts.size() + " entradas" + extension;
                    save(json, file, folder);
                    System.out.println(Thread.currentThread().getName() + ": Número de post guardados: " + listPosts.size());
                    listPosts.clear();
                } catch (JsonProcessingException e) {
                    e.printStackTrace();
                }
            }
            // System.out.println(Thread.currentThread().getName() + ": Hilo Procesado " + idThread);
        }
        System.out.println(Thread.currentThread().getName() + ": Bloque finalizado " + endThread);
    }

    private String buildFullUrl(int idThread, int pag) {
        return urlBase +
                "?" +
                "t=" +
                idThread +
                "&" +
                "page=" +
                pag;
    }

    private int getNumPags(Document doc) {
        int ret = 0;
        try {
            Element container = doc.getElementById("container");
            Elements pagsLinks = container.getElementsByTag("section").get(1).getElementsByTag("a");
            if (!pagsLinks.isEmpty()) {
                String pag = pagsLinks.last().text();
                System.out.println(Thread.currentThread().getName() + ": Numero de paginas en el hilo: " + pag);
                ret = Integer.parseInt(pag);
            } else {
                // System.out.println(Thread.currentThread().getName() + ": ERROR: Error buscando links de páginas del hilo, pocas pagina");
            }
        } catch (NumberFormatException e) {
            System.out.println(Thread.currentThread().getName() + ": ERROR: Error al pasar a entero el número de páginas del hilo");
        } catch (Exception e) {
            System.out.println(Thread.currentThread().getName() + ": ERROR: Error leyendo número de páginas del hilo");
        }
        return ret;
    }

    private Post extractPost(Element element) throws Exception {
        // printAtributos(element);
        Post post = new Post();
        // if (!element.getElementsByTag("h1").text().equals("")) {
        // hilo = element.getElementsByTag("h1").text();
        // }
        post.setUrl(fullUrl);
        post.setHilo(nameHilo);
        post.setId(element.id());
        post.setNombreUsuario(extractNombreUsuario(element));
        post.setFecha(extractDate(element));
        post.setNumeroEntrada(extractNumeroEntrada(element));
        post.setTipoUsuario(extractTipoUsuario(element));
        Cita cita = extractCita(element);
        post.setCita(cita != null ? cita.getIdPost() : null);
        post.setContenido(extractContenido(element, cita));
        // System.out.println("\tPost: " + post.getNumeroEntrada());
        return post;
    }

    private Cita extractCita(Element element) throws Exception {
        Elements e = element.getElementsByClass("quote");
        if (e.isEmpty())
            return null;

        if (e.size() > 1)
            throw new Exception("Mas de una cita");

        Cita cita = new Cita();
        cita.setText(e.first().text());
        Elements as = e.first().getElementsByTag("a");

        if (!as.isEmpty()) {
            Element a = as.first();
            // a.attributes().get("href")
            StringBuilder sb = new StringBuilder(a.attributes().get("href"));
            int i = sb.lastIndexOf("#");
            sb.delete(0, i + 1);
            cita.setIdPost(sb.toString());
        } else {
            cita.setIdPost("unknow");
        }
        return cita;
    }

    // private String extractContenido(Element element, Cita cita) {
    //     String strCita = cita == null ? "" : cita.getText();
    //     String id = new StringBuilder(element.id()).insert(4, "_message_").toString();
    //     // System.out.println(id);
    //     Element e = element.getElementById(id);
    //     // printAtributos(e);
    //     StringBuilder sb = new StringBuilder(e.text());
    //     sb.delete(0, strCita.length() + (cita == null ? 0 : 1));
    //     return sb.toString();
    // }

    private String extractContenido(Element element, Cita cita) {
        String id = new StringBuilder(element.id()).insert(4, "_message_").toString();
        Element e = element.getElementById(id);
        String ret;
        if (cita != null) {
            int ini = e.text().indexOf(cita.getText());
            int fin = ini + cita.getText().length() + 1;
            ret = new StringBuilder(e.text()).delete(ini, fin).toString();
            // String patron = cita.getText().replaceAll("?", "\\?");
            // ret = e.text().replaceAll(cita.getText() + " ", "");
        } else {
            ret = e.text();
        }

        return ret;
    }

    private String extractTipoUsuario(Element element) {
        Elements e = element.getElementsByClass("subtitle-small-gray");
        // printAtributos(e.first());
        return e.first().text();
    }

    private int extractNumeroEntrada(Element element) {
        Elements e = element.getElementsByClass("date-and-time-gray");
        StringBuilder eb = new StringBuilder(e.get(1).text());
        eb.deleteCharAt(0);
        // printAtributos(e.last());
        return Integer.valueOf(eb.toString());
    }

    private String extractDate(Element element) {
        Elements e = element.getElementsByClass("postdate old");
        // printAtributos(e.first());
        return e.first().text();
    }

    private String extractNombreUsuario(Element element) {
        String id = new StringBuilder(element.id()).insert(4, "menu_").toString();
        Element e = element.getElementById(id);
        // printAtributos(e);
        return e.text();
    }

    private void printAtributos(Element element) {
        System.out.println("######################### Hilo #########################");
        System.out.println("Titulo Hilo: " + nameHilo);
        System.out.println("######################### text #########################");
        System.out.println(element.text());
        System.out.println("######################### toString #########################");
        System.out.println(element);
        System.out.println("######################### data #########################");
        System.out.println(element.data());
        System.out.println("######################### outerHtml #########################");
        System.out.println(element.outerHtml());
        System.out.println("######################### ownText #########################");
        System.out.println(element.ownText());
        System.out.println("######################### tagName #########################");
        System.out.println(element.tagName());
        System.out.println("######################### val #########################");
        System.out.println(element.val());
        System.out.println("######################### tag #########################");
        System.out.println(element.tag());
        System.out.println("######################### nodeName #########################");
        System.out.println(element.nodeName());
        System.out.println("######################### id #########################");
        System.out.println(element.id());
        System.out.println("######################### html #########################");
        System.out.println(element.html());
        System.out.println("######################### dataset #########################");
        System.out.println(element.dataset());
        System.out.println("######################### element #########################");
        System.out.println(element);
    }

    private String scrap(String url, String fileName, String folder) throws Exception {
        String content = getRemoteContents(url);
        save(content, fileName, folder);
        System.out.println(content);
        return content;
    }

    private String scrap(String url, String fileName, String folder, int pags) throws Exception {
        folder = "./html/";
        pags = 125;
        String content = null;
        for (int i = 1; i <= pags; i++) {
            fileName = "pag" + i + ".html";
            content = getRemoteContents("https://www.htcmania.com/showthread.php?t=1096138&page=" + i);
            save(content, fileName, folder);
            System.out.println(content);
        }
        return content;
    }

    private String getRemoteContents(String url) throws IOException {
        URL urlObject = new URL(url);
        URLConnection conn = urlObject.openConnection();
        conn.setRequestProperty("User-Agent",
                "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10.4; en-US; rv:1.9.2.2) Gecko/20100316 Firefox/3.6.2");
        BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        String inputLine, output = "";
        while ((inputLine = in.readLine()) != null) {
            output += inputLine;
        }
        in.close();

        return output;
    }

    private void patron(String input) throws Exception {
        // <div id="edit\w+</div>
        String regex = "<div id=\"edit.+</div>";
        Pattern pat = Pattern.compile(regex);
        Matcher mat = pat.matcher(input);
        if (mat.find()) {
            mat.group(0);
            // ........
        }
    }

    private void save(List<Post> listPosts, String folder) {
        ObjectMapper mapper = new ObjectMapper();
        String json;
        try {
            json = mapper.writeValueAsString(listPosts);

            File fileWrite = new File(folder + "/tmp.json");
            FileWriter writer = new FileWriter(fileWrite, true);
            System.out.println("Escribir fichero csv...");
            writer.write(json);
            System.out.println("Fichero csv salvado");
            writer.close();
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Número de post guardados: " + listPosts.size());
    }

    private void save(String textToBeWritten, String fileName, String folder) {
        fileName = cleanFileName(fileName);
        String filePath = folder + fileName;
        PrintWriter printWriter = null;
        mkdir(folder);
        try {
            // printWriter = new PrintWriter(filePath, StandardCharsets.UTF_8);
            // printWriter = new PrintWriter(filePath);
            printWriter = new PrintWriter(filePath, StandardCharsets.UTF_8);
        } catch (FileNotFoundException e) {
            System.out.println("Unable to locate the fileName: " + e.getMessage());
        } catch (IOException e) {
            System.out.println("Charset no soportado");
            e.printStackTrace();
        }
        Objects.requireNonNull(printWriter).println(textToBeWritten);
        printWriter.close();
    }

    private String cleanFileName(String fileName) {
        fileName = fileName.replaceAll("[?¿!¡:;/\\\\\"']", "");
        return fileName;
    }

    private void mkdir(String path) {
        File directorio = new File(path);
        if (!directorio.exists()) {
            if (directorio.mkdirs()) {
                System.out.println("Directorio creado");
            } else {
                System.out.println("Error al crear directorio");
            }
        }
    }

    // /**
    // * Convierte un fichero json en un fichero csv
    // *
    // * @param jsonPath Ruta del fichero origen
    // * @param csvPath Ruta del fichero destino
    // */
    // public void jsonToCsv(String jsonPath, String csvPath) {
    // try {
    // // Mapear fichero JSON
    // JsonNode jsonTree = new ObjectMapper().readTree(new File(jsonPath));

    // // Recuperar primer elemento
    // JsonNode firstObject = jsonTree.elements().next();

    // StringBuilder csv = new StringBuilder();

    // // Añadir cabecera de la tabla
    // firstObject.fields().forEachRemaining(f -> addHeadColumn(csv, f));
    // csv.append("\n");

    // // Añadir primera fila
    // firstObject.fields().forEachRemaining(f -> addCel(csv, f));
    // csv.append("\n");

    // // Añadir el resto de filas
    // jsonTree.forEach(j -> addRows(csv, j));

    // // Salvar fichero csv
    // File fileWrite = new File(csvPath);
    // FileWriter writer = new FileWriter(fileWrite, false);
    // System.out.println("Escribir fichero csv...");
    // writer.write(csv.toString());
    // System.out.println("Fichero csv salvado");
    // writer.close();
    // } catch (IOException e) {
    // e.printStackTrace();
    // }
    // }

    // /**
    // * Añade una fila al StringBuilder
    // *
    // * @param csv StringBuilder al que se añade la fila
    // * @param j Objeto JSON con los datos de la fila a añadir
    // */
    // private void addRows(StringBuilder csv, JsonNode j) {
    // j.fields().forEachRemaining(f -> addCel(csv, f));
    // csv.append("\n");
    // }

    // /**
    // * Añade un campo al StringBuilder
    // *
    // * @param csv StringBuilder al que se añade la fila
    // * @param f Par clave valor con el contenido del campo
    // */
    // private void addCel(StringBuilder csv, Entry<String, JsonNode> f) {
    // csv.append(f.getValue()).append(";");
    // System.out.println(f.getValue());
    // }

    // /**
    // * Añade la cabecera de una columna de la tabla.
    // *
    // * @param csv StringBuilder al que se añade la fila
    // * @param f Par clave valor con el contenido de la cabecera
    // */
    // private void addHeadColumn(StringBuilder csv, Entry<String, JsonNode> f) {
    // csv.append(f.getKey()).append(";");
    // }

    // /**
    // * NO FUNCIONA
    // *
    // * @param jsonPath
    // * @param csvPath
    // * @throws IOException
    // */
    // public void jsonToCsv(String jsonPath, String csvPath) throws IOException {
    // JsonNode jsonTree = new ObjectMapper().readTree(new File(jsonPath));

    // final com.fasterxml.jackson.dataformat.csv.CsvSchema.Builder csvSchemaBuilder
    // = CsvSchema.builder();
    // JsonNode firstObject = jsonTree.elements().next();

    // firstObject.fields().forEachRemaining(f -> extracted(csvSchemaBuilder, f));

    // // while (firstObject.fields().hasNext()) {
    // // Entry<String, JsonNode> f = firstObject.fields().next();
    // // csvSchemaBuilder.addColumn(f.getKey());
    // // firstObject.fields().remove();
    // // }
    // CsvSchema csvSchema = csvSchemaBuilder.build().withHeader();

    // CsvMapper csvMapper = new CsvMapper();
    // csvMapper.writerFor(JsonNode.class)
    // .with(csvSchema)

    // .writeValue(new File(csvPath), jsonTree);
    // }

    // private com.fasterxml.jackson.dataformat.csv.CsvSchema.Builder extracted(
    // final com.fasterxml.jackson.dataformat.csv.CsvSchema.Builder
    // csvSchemaBuilder, Entry<String, JsonNode> f) {
    // return csvSchemaBuilder.addColumn(f.getKey());
    // }

}
