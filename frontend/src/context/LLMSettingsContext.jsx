import React, { createContext, useContext, useState } from 'react'

const LLMSettingsContext = createContext(null)

export const LLMSettingsProvider = ({ children }) => {
  const [provider, setProvider] = useState('openai')
  const [model, setModel] = useState('gpt-4o-mini')

  const value = {
    provider,
    model,
    setProvider,
    setModel,
  }

  return (
    <LLMSettingsContext.Provider value={value}>
      {children}
    </LLMSettingsContext.Provider>
  )
}

export const useLLMSettings = () => {
  const ctx = useContext(LLMSettingsContext)
  if (!ctx) {
    throw new Error('useLLMSettings must be used within LLMSettingsProvider')
  }
  return ctx
}
