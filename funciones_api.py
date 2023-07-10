from fastapi import FastAPI    
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('movies_dataset_new.csv') #traemos el dataset
app = FastAPI()   # Instanciamos FastAPI


@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    '''Ingresas el idioma, retornando la cantidad de peliculas producidas en el mismo'''
    respuesta = df[df['original_language'] == idioma].shape[0]
    
    return {'idioma':idioma, 'cantidad':respuesta}


@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula:str):
    '''Ingresas la pelicula, retornando la duracion y el año'''
    pelicula_info = df[df['title'] == pelicula]
    duracion = pelicula_info['runtime'].values[0]
    anio = pelicula_info['release_year'].values[0]
    respuesta = int(duracion)
    anio_respuesta = int(anio)
    
    return {'pelicula':pelicula, 'duracion':respuesta, 'anio':anio_respuesta}


@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
    franquicia_movies = df[df['belongs_to_collection'] == franquicia]
    cantidad_peliculas = len(franquicia_movies)
    ganancia_total = franquicia_movies['revenue'].sum()
    ganancia_promedio = franquicia_movies['revenue'].mean()
    
    return {'franquicia':franquicia, 'cantidad':cantidad_peliculas, 'ganancia_total':ganancia_total, 'ganancia_promedio':ganancia_promedio}


@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''
    pais_movies = df[df['production_countries'].apply(lambda paises: pais in paises)]
    cantidad_peliculas = len(pais_movies)
    respuesta = cantidad_peliculas
    
    return {'pais':pais, 'cantidad':respuesta}


@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    '''Ingresas la productora, entregandote el revunue total y la cantidad de peliculas que realizo '''
    productoras_movies = df[df['production_companies'].apply(lambda productoras: productora in productoras)]
    revenue_total = productoras_movies['revenue'].sum()
    cantidad_peliculas = len(productoras_movies)

    return {'productora':productora, 'revenue_total': revenue_total,'cantidad':cantidad_peliculas}


@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    ''' Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma. En formato lista'''
    director_movies = df[df['director'] == nombre_director]
    result = []
    total_return = 0
    for _, row in director_movies.iterrows():
        roi = row['ROI']
        movie_info = {
            'pelicula': row['title'],
            'fecha_lanzamiento': row['release_date'],
            'retorno': roi,
            'costo': row['budget'],
            'ganancia': row['revenue'] - row['budget']
        }
        result.append(movie_info)
        if roi > 0.0:
            total_return += roi
    
    return {'director':nombre_director, 'retorno_total_director':total_return, 'peliculas': result}

# ML
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''  
    titulo = titulo.lower().strip()
    pelicula = df[df['title'].str.lower().str.strip() == titulo]
    generos = df['genres'].str.get_dummies(',')
    puntuaciones = df[['vote_average', 'vote_count']].values
    caracteristicas = pd.concat([generos, pd.DataFrame(puntuaciones, columns=['vote_average', 'vote_count'])], axis=1)
    similitudes = cosine_similarity(caracteristicas.loc[pelicula.index], caracteristicas)
    indices_similares = similitudes.argsort()[0][::-1][1:]
    peliculas_similares = df.iloc[indices_similares][:5]['title'].tolist()
    
    return {'lista_recomendada': peliculas_similares}