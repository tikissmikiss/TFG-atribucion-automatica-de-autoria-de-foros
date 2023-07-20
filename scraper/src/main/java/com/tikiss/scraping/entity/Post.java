package com.tikiss.scraping.entity;

import lombok.Data;

@Data
public class Post {

    private String id;
    private String nombreUsuario;
    private String tipoUsuario;
    private String fecha;
    private int numeroEntrada;
    private String hilo;
    private String cita;
    private String contenido;
    private String url;

}
