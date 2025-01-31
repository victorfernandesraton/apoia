// Nível de sigilo a ser aplicado ao processo. Dever-se-á utilizar os seguintes níveis:
// - 0: públicos, acessíveis a todos os servidores do Judiciário e dos demais órgãos públicos de colaboração na administração da Justiça, assim como aos advogados e a qualquer cidadão
// - 1: segredo de justiça, acessíveis aos servidores do Judiciário, aos servidores dos órgãos públicos de colaboração na administração da Justiça e às partes do processo.
// - 2: sigilo mínimo, acessível aos servidores do Judiciário e aos demais órgãos públicos de colaboração na administração da Justiça 
// - 3: sigilo médio, acessível aos servidores do órgão em que tramita o processo, à(s) parte(s) que provocou(ram) o incidente e àqueles que forem expressamente incluídos 
// - 4: sigilo intenso, acessível a classes de servidores qualificados (magistrado, diretor de secretaria/escrivão, oficial de gabinete/assessor) do órgão em que tramita o processo, às partes que provocaram o incidente e àqueles que forem expressamente incluídos 
// - 5: sigilo absoluto, acessível apenas ao magistrado do órgão em que tramita, aos servidores e demais usuários por ele indicado e às partes que provocaram o incidente.
export const assertNivelDeSigilo = (nivel, descrDaPeca?) => {
    const nivelMax = parseInt(process.env.CONFIDENTIALITY_LEVEL_MAX as string)
    nivel = parseInt(nivel)
    if (nivel > nivelMax)
        throw new Error(`Nível de sigilo '${nivel}'${descrDaPeca ? ' da peça ' + descrDaPeca : ''} maior que o máximo permitido '${nivelMax}'.`)
}

export const verificarNivelDeSigilo = () => {
    return !(process.env.CONFIDENTIALITY_LEVEL_MAX === undefined || process.env.CONFIDENTIALITY_LEVEL_MAX === '')
}

