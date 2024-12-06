import { createOpenAI } from '@ai-sdk/openai'
import { createAnthropic } from '@ai-sdk/anthropic'
import { createGoogleGenerativeAI } from '@ai-sdk/google'
import { getPrefs } from '../utils/prefs'
import { LanguageModelV1 } from '@ai-sdk/provider'
import { createGroq } from '@ai-sdk/groq'
import { createAzure } from "@ai-sdk/azure"
import { EMPTY_PREFS_COOKIE, PrefsCookieType } from '@/lib/utils/prefs-types';
import { ModelProvider } from './model-types'

export function getKeysByModel(model: string): string[] {
    if (model.startsWith('claude-')) {
        return [ModelProvider.ANTHROPIC.apiKey]
    } else if (model.startsWith('gpt-')) {
        return [ModelProvider.OPENAI.apiKey, ModelProvider.AZURE.apiKey]
    } else if (model.startsWith('gemini-')) {
        return [ModelProvider.GOOGLE.apiKey]
    } else if (model.startsWith('llama-')) {
        return [ModelProvider.GROQ.apiKey]
    }
}

export function getAvaliableKeysAPI(prefs: PrefsCookieType): Map<string, string> {
    const keys = Object.values(ModelProvider).map(provider => provider.apiKey)
    let result = new Map<string, string>()
    for (const [key, value] of Object.entries(process.env)) {
        if (keys.includes(key)) {
            result.set(key, value)
        }
    }
    for (const [key, value] of Object.entries(prefs.env)) {
        if (keys.includes(key)) {
            result.set(key, value)
        }
    }
    return result
}

export function getModel(params?: { structuredOutputs: boolean, overrideModel?: string }): { model: string, modelRef: LanguageModelV1 } {
    const prefs = getPrefs()
    let model: string
    if (prefs) {
        model = prefs.model
    } else {
        model = process.env.MODEL as string
    }

    if (params?.overrideModel) model = params.overrideModel
    const apiKeysMap = getAvaliableKeysAPI(prefs || EMPTY_PREFS_COOKIE)
    const keysByModel = getKeysByModel(model)
    let apiKey: string
    let apiType: string
    for (const key of keysByModel) {
        if (apiKeysMap.has(key)) {
            apiKey = apiKeysMap.get(key)
            apiType = key
            break
        }
    }
    switch (apiType) {
        case ModelProvider.ANTHROPIC.apiKey:
            const anthropic = createAnthropic({ apiKey })
            return { model, modelRef: anthropic(model) }
        case ModelProvider.OPENAI.apiKey:
            const openai = createOpenAI({ apiKey })
            return { model, modelRef: openai(model, { structuredOutputs: params?.structuredOutputs }) }
        case ModelProvider.GOOGLE.apiKey:
            const google = createGoogleGenerativeAI({ apiKey })
            return { model, modelRef: google(model, { structuredOutputs: params?.structuredOutputs }) }
        case ModelProvider.GROQ.apiKey:
            const groq = createGroq({ apiKey })
            return { model, modelRef: groq(model, {}) }
        case ModelProvider.AZURE.apiKey:
            const azure = createAzure({
                apiKey: apiKey,
                // TODO: mover isso para ser editado tanbém nas configs de models
                resourceName: process.env.AZURE_RESOURCE,
                // TODO: mover isso para ser editado tanbém nas configs de models
                apiVersion: "2024-02-15-preview"
            })
            return { model, modelRef: azure(model, { structuredOutputs: params.structuredOutputs }) }
        default:
            throw new Error(`Model ${model} not found`)
    }
}

