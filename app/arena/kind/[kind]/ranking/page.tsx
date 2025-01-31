import { Container } from 'react-bootstrap'
import RankingContents from './ranking-contents'


export default async function New({ params }: { params: { kind: string } }) {
    return <Container fluid={false}>
        <h1 className="mt-5 mb-3">Ranking de Prompts</h1><RankingContents kind={params.kind} />
    </Container>
}